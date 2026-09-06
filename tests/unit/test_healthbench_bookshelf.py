"""Offline official-protocol shapes with synthetic text; no model or HTTP."""

from __future__ import annotations

import gzip
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
import zlib

from src.interactive.healthbench_bookshelf import (
    BOOKSHELF_COLLECTION_FILTERS,
    BOOKSHELF_MAX_RESPONSE_BYTES,
    BOOKSHELF_OAI_BASE_URL,
    BOOKSHELF_SOURCE,
    BOOKSHELF_SOURCE_TYPE,
    BookshelfClient,
)
from src.interactive.healthbench_clinical_tools import _source_page


def summary(**overrides):
    return {
        "uid": "901", "title": "Matched subsection", "accessionid": "NBK123",
        "pubdate": "2026/01/02 00:00", "book": "pdqcis", "rtype": "sec",
        "bookinfo": '<Info><Path><Parent role="source"><Title>Synthetic book</Title></Parent>'
        '<Parent role="document"><Title>Synthetic chapter</Title></Parent></Path></Info>',
        "text": "Unverified summary text must not become full text.", **overrides,
    }


def oai(document, *, identifier="123", sets="pdqcis", prefix="nbk_ftext"):
    return (f'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
            f'<request metadataPrefix="{prefix}"/><GetRecord><record><header>'
            f'<identifier>oai:books.ncbi.nlm.nih.gov:{identifier}</identifier>'
            f'<datestamp>2026-02-03</datestamp><setSpec>{sets}</setSpec><setSpec>book-open</setSpec>'
            f'</header><metadata>{document}</metadata></record></GetRecord></OAI-PMH>').encode()


BITS_DOCUMENT = '''<book-part-wrapper xmlns="https://jats.nlm.nih.gov/ns/extensions/bits/2.0/">
<book-meta><book-title-group><book-title>Synthetic book</book-title></book-title-group>
<pub-date date-type="pubr"><year>2002</year></pub-date></book-meta>
<book-part><book-part-meta><book-part-id book-part-id-type="art-access-id">NBK123</book-part-id>
<title-group><title>Synthetic chapter</title><subtitle>Health Professional Version</subtitle></title-group>
<pub-date date-type="pub"><day>02</day><month>01</month><year>2026</year></pub-date>
<abstract><p>Abstract alone is not the body.</p></abstract></book-part-meta><body>
<sec><title>Results</title><p>Clinical <italic>condition</italic> retained.</p>
<sec><title>Limitations</title><p>Important uncertainty.</p></sec>
<table-wrap><caption>Fixture table</caption><table><tr><td>42</td><td>units</td></tr></table></table-wrap>
<fig><caption>Image caption</caption><alt-text>Image description</alt-text><graphic/></fig></sec>
</body><back><ref-list><ref>Source citation.</ref></ref-list></back></book-part></book-part-wrapper>'''

LEGACY_DOCUMENT = '''<book-part xmlns="https://dtd.nlm.nih.gov/ns/book/2.3/"
xmlns:xlink="http://www.w3.org/1999/xlink" book-part-type="appendix">
<book-meta><book-title-group><book-title>Synthetic AHRQ report</book-title></book-title-group>
<edition>Second edition</edition><pub-date pub-type="epub"><month>04</month><year>2015</year></pub-date>
<notes><p><related-object link-type="collection-link" source-id="hscompeffcollect">AHRQ Comparative Effectiveness Reviews</related-object></p></notes></book-meta>
<book-part-meta><title-group><title content-type="book">Synthetic AHRQ report</title>
<subtitle content-type="appendix">Appendix G: Methods</subtitle></title-group>
<uri xlink:href="art-access-id://NBK123"/></book-part-meta>
<body><fig><caption><title>Methods figure</title></caption><alt-text>Actual diagram description.</alt-text>
<graphic xlink:href="figure-file"/></fig></body></book-part>'''


class BytesResponse:
    def __init__(self, payload, encoding=""):
        self.payload, self.closed = payload, False
        self.headers = {"Content-Encoding": encoding}

    def read(self, size):
        return self.payload[:size] if isinstance(self.payload, bytes) else self.payload

    def close(self):
        self.closed = True


class FixtureOpener:
    def __init__(self, rows=None, ids=None, xml=None, error=None, encoding=""):
        self.rows = rows if rows is not None else {"901": summary()}
        self.ids = ids if ids is not None else list(self.rows)
        self.xml = xml if xml is not None else oai(BITS_DOCUMENT)
        self.error, self.encoding = error, encoding
        self.calls, self.responses = [], []

    def __call__(self, request, *, timeout):
        self.calls.append((request, timeout))
        if self.error:
            raise self.error
        path = urlparse(request.full_url).path
        if path.endswith("esearch.fcgi"):
            payload = json.dumps({"esearchresult": {"idlist": self.ids}}).encode()
        elif path.endswith("esummary.fcgi"):
            payload = json.dumps({"result": {"uids": list(self.rows), **self.rows}}).encode()
        else:
            payload = self.xml
        response = BytesResponse(payload, self.encoding if not path.endswith(".fcgi") else "")
        self.responses.append(response)
        return response


class BookshelfClientTests(unittest.TestCase):
    def setUp(self):
        rate_limit = patch("src.interactive.healthbench_bookshelf._wait_for_pubmed_request_slot")
        self.rate_limit = rate_limit.start()
        self.addCleanup(rate_limit.stop)

    def test_metadata_search_uses_uid_mapping_and_never_claims_full_text(self):
        opener = FixtureOpener()
        result = BookshelfClient(opener=opener).search("synthetic clinical query")
        item = result["evidence"][0]
        self.assertEqual(result["operation"], "search")
        self.assertEqual(result["query"], "synthetic clinical query")
        self.assertEqual(item["source"], BOOKSHELF_SOURCE)
        self.assertEqual(item["source_type"], BOOKSHELF_SOURCE_TYPE)
        self.assertEqual(item["source_id"], "bookshelf:NBK123")
        self.assertNotEqual(item["document_id"], "901")
        self.assertEqual(item["title"], "Synthetic chapter")
        self.assertEqual(item["matched_title"], "Matched subsection")
        self.assertEqual(item["book_title"], "Synthetic book")
        self.assertEqual(item["date"], "2026/01/02 00:00")
        self.assertIsNone(item["version"])
        self.assertEqual(item["url"], "https://www.ncbi.nlm.nih.gov/books/NBK123/")
        self.assertEqual(item["content_type"], "bookshelf_metadata_not_full_text")
        self.assertEqual(item["excerpt"], "")
        self.assertEqual(item["full_text_availability"], "unverified_until_source_read")
        self.assertEqual(result["source_receipts"][0]["result_count"], 1)
        self.assertEqual(len(opener.calls), 2)
        for request, timeout in opener.calls:
            self.assertEqual(timeout, 8.0)
            self.assertEqual(parse_qs(urlparse(request.full_url).query)["db"], ["books"])
        self.assertTrue(all(response.closed for response in opener.responses))
        search_params = parse_qs(urlparse(opener.calls[0][0].full_url).query)
        self.assertEqual(search_params["retmax"], ["3"])
        self.assertEqual(search_params["sort"], ["relevance"])
        self.assertEqual(self.rate_limit.call_count, 2)

    def test_official_collection_scopes_preserve_the_query(self):
        for collection in ("bookshelf", "pdq", "ahrq"):
            with self.subTest(collection=collection):
                opener = FixtureOpener()
                result = BookshelfClient(collection=collection, opener=opener).search("synthetic therapy")
                term = parse_qs(urlparse(opener.calls[0][0].full_url).query)["term"][0]
                scope = BOOKSHELF_COLLECTION_FILTERS[collection]
                self.assertEqual(term, f"(synthetic therapy) AND {scope}" if scope else "synthetic therapy")
                self.assertEqual(result["search_collection"], collection)
                self.assertEqual(result["evidence"][0]["source"], BOOKSHELF_SOURCE)

    def test_search_bounded_deduplication_preserves_server_order(self):
        rows = {"901": summary(), "902": summary(uid="902"), "903": summary(uid="903", accessionid="NBK456")}
        opener = FixtureOpener(rows=rows, ids=["901", "902", "903", "904"])
        items = BookshelfClient(opener=opener).search("synthetic study")["evidence"]
        self.assertEqual([item["source_id"] for item in items], ["bookshelf:NBK123", "bookshelf:NBK456"])
        self.assertEqual([item["rank"] for item in items], [1, 2])
        self.assertEqual(parse_qs(urlparse(opener.calls[1][0].full_url).query)["id"], ["901,902,903"])

    def test_empty_search_makes_one_request_with_successful_zero_receipt(self):
        opener = FixtureOpener(rows={})
        result = BookshelfClient(opener=opener).search("synthetic absent")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["source_receipts"][0]["result_count"], 0)
        self.assertEqual(len(opener.calls), 1)

    def test_search_rejects_invalid_ids_or_mismatched_summaries(self):
        for opener in (FixtureOpener(ids=["NBK123"]), FixtureOpener(rows={"901": summary(uid="902")}), FixtureOpener(rows={"901": summary(accessionid="901")})):
            with self.subTest(opener=opener), self.assertRaises(RuntimeError):
                BookshelfClient(opener=opener).search("synthetic study")

    def test_bits_read_retains_actual_text_and_metadata(self):
        opener = FixtureOpener()
        item = BookshelfClient(opener=opener).read("bookshelf:NBK123")
        self.assertEqual(item["title"], "Synthetic chapter: Health Professional Version")
        self.assertEqual(item["date"], "2026-01-02")
        self.assertEqual(item["repository_datestamp"], "2026-02-03")
        self.assertIsNone(item["version"])
        self.assertEqual(item["collection"], "pdq")
        self.assertEqual(item["text_scope"], "book_part")
        self.assertEqual(item["content_type"], "bookshelf_open_access_xml_text")
        self.assertEqual(item["source"], BOOKSHELF_SOURCE)
        for words in ("Clinical condition retained.", "Results", "Limitations", "Important uncertainty.", "42", "units", "Image caption", "Image description", "Source citation.", "Abstract alone is not the body."):
            self.assertIn(words, item["excerpt"])
        self.assertEqual(item["excerpt"].count("Important uncertainty."), 1)
        self.assertEqual(len(opener.calls), 1)
        request, timeout = opener.calls[0]
        self.assertTrue(request.full_url.startswith(BOOKSHELF_OAI_BASE_URL))
        params = parse_qs(urlparse(request.full_url).query)
        self.assertEqual(params["verb"], ["GetRecord"])
        self.assertEqual(params["metadataPrefix"], ["nbk_ftext"])
        self.assertEqual(params["identifier"], ["oai:books.ncbi.nlm.nih.gov:123"])
        self.assertEqual(timeout, 8.0)
        self.assertEqual(request.get_header("Accept-encoding"), "gzip, deflate")
        self.rate_limit.assert_not_called()

    def test_legacy_ahrq_xml_and_available_caption_alt_text(self):
        item = BookshelfClient(opener=FixtureOpener(xml=oai(LEGACY_DOCUMENT, sets="cer151"))).read("bookshelf:NBK123")
        self.assertEqual(item["collection"], "ahrq")
        self.assertEqual(item["date"], "2015-04")
        self.assertEqual(item["version"], "Second edition")
        self.assertIn("Appendix G: Methods", item["title"])
        self.assertIn("Actual diagram description.", item["excerpt"])
        self.assertEqual(item["source"], BOOKSHELF_SOURCE)

    def test_book_xml_projects_body_and_back_matter(self):
        doc = '<book><book-meta><book-title-group><book-title>Synthetic whole book</book-title></book-title-group></book-meta><book-body><book-part><body><p>Book chapter body.</p></body></book-part></book-body><book-back><p>Book references.</p></book-back></book>'
        item = BookshelfClient(opener=FixtureOpener(xml=oai(doc, sets="fixture"))).read("bookshelf:NBK123")
        self.assertEqual(item["text_scope"], "book")
        self.assertEqual(item["collection"], "bookshelf")
        self.assertIsNone(item["date"])
        self.assertIn("Book chapter body.", item["excerpt"])
        self.assertIn("Book references.", item["excerpt"])

    def test_pagination_is_caller_owned_without_silent_truncation(self):
        xml = oai(BITS_DOCUMENT.replace("Important uncertainty.", "Important uncertainty. " * 1500))
        item = BookshelfClient(opener=FixtureOpener(xml=xml)).read("bookshelf:NBK123")
        first = _source_page(item)
        second = _source_page(item, first["next_offset"])
        third = _source_page(item, second["next_offset"]) if second["truncated"] else None
        restored = first["excerpt"] + second["excerpt"] + (third["excerpt"] if third else "")
        self.assertEqual(restored, item["excerpt"])
        self.assertTrue(first["truncated"])
        self.assertNotIn("offset", item)

    def test_no_body_or_metadata_only_never_becomes_full_text(self):
        missing_body = BITS_DOCUMENT[:BITS_DOCUMENT.index("<body>")] + "</book-part></book-part-wrapper>"
        fixtures = (oai(missing_body), oai(BITS_DOCUMENT, prefix="nbk_meta"))
        for xml in fixtures:
            with self.subTest(xml=xml[:80]), self.assertRaisesRegex(RuntimeError, "metadata|body"):
                BookshelfClient(opener=FixtureOpener(xml=xml)).read("bookshelf:NBK123")

    def test_permission_unavailable_raises_without_fallback_or_retry(self):
        xml = b'<OAI-PMH><error code="cannotDisseminateFormat">Full text is unavailable.</error></OAI-PMH>'
        opener = FixtureOpener(xml=xml)
        with self.assertRaisesRegex(LookupError, "cannotDisseminateFormat"):
            BookshelfClient(opener=opener).read("bookshelf:NBK123")
        self.assertEqual(len(opener.calls), 1)

    def test_read_rejects_wrong_identity_deleted_and_non_book_payloads(self):
        fixtures = (
            oai(BITS_DOCUMENT, identifier="999"),
            oai(BITS_DOCUMENT.replace("NBK123", "NBK999")),
            oai(LEGACY_DOCUMENT.replace("NBK123", "NBK999")),
            oai(BITS_DOCUMENT).replace(b"<header>", b'<header status="deleted">'),
            oai("<article><body><p>Wrong format.</p></body></article>"),
            b"<html><body>A web page is not OAI XML.</body></html>",
        )
        for xml in fixtures:
            with self.subTest(xml=xml[:80]), self.assertRaises(RuntimeError):
                BookshelfClient(opener=FixtureOpener(xml=xml)).read("bookshelf:NBK123")

    def test_invalid_source_ids_fail_before_network(self):
        opener = FixtureOpener()
        for source_id in (None, "NBK123", "bookshelf:123", "bookshelf:NBK123.2", "bookshelf:NBK123/section", "bookshelf:NBK123 "):
            with self.subTest(source_id=source_id), self.assertRaises(ValueError):
                BookshelfClient(opener=opener).read(source_id)
        self.assertEqual(opener.calls, [])

    def test_query_limits_are_reused_before_network(self):
        opener = FixtureOpener()
        for query in (None, "", " ", "synthetic " * 500):
            with self.subTest(query=query), self.assertRaises(ValueError):
                BookshelfClient(opener=opener).search(query)
        self.assertEqual(opener.calls, [])

    def test_gzip_and_deflate_official_transport(self):
        xml = oai(BITS_DOCUMENT)
        for encoding, payload in (("gzip", gzip.compress(xml)), ("deflate", zlib.compress(xml))):
            with self.subTest(encoding=encoding):
                item = BookshelfClient(opener=FixtureOpener(xml=payload, encoding=encoding)).read("bookshelf:NBK123")
                self.assertIn("Clinical condition retained.", item["excerpt"])

    def test_wire_and_decoded_size_limit_rejects_not_truncates(self):
        oversized = b"x" * (BOOKSHELF_MAX_RESPONSE_BYTES + 1)
        for encoding, payload in (("", oversized), ("gzip", gzip.compress(oversized)), ("deflate", zlib.compress(oversized))):
            with self.subTest(encoding=encoding), self.assertRaisesRegex(RuntimeError, "bounded document size"):
                BookshelfClient(opener=FixtureOpener(xml=payload, encoding=encoding)).read("bookshelf:NBK123")

    def test_transport_failure_no_retry(self):
        for method, query in (("search", "synthetic study"), ("read", "bookshelf:NBK123")):
            opener = FixtureOpener(error=HTTPError("https://example.invalid", 503, "unavailable", {}, None))
            with self.assertRaises(HTTPError):
                getattr(BookshelfClient(opener=opener), method)(query)
            self.assertEqual(len(opener.calls), 1)

    def test_constructor_bounds(self):
        for arguments in ({"timeout_seconds": 0}, {"timeout_seconds": float("inf")}, {"timeout_seconds": True}, {"retmax": 0}, {"retmax": 4}, {"retmax": True}, {"collection": "all_ahrq"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                BookshelfClient(**arguments)
        with self.assertRaises(TypeError):
            BookshelfClient(opener=None)


if __name__ == "__main__":
    unittest.main()
