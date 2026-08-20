from pipeline.scrape import collect_info_links, filename_to_url, url_to_filename

SAMPLE_HTML = """
<html><body>
<a href="/info/buyer-protection.htm">Käuferschutz</a>
<a href="https://www.chrono24.de/info/trusted-checkout.htm?x=1#top">Trusted Checkout</a>
<a href="/info/buyer-protection.htm">Duplikat</a>
<a href="/watches/rolex.htm">keine Info-Seite</a>
<a href="https://example.com/info/fremd.htm">fremde Domain</a>
<a href="/info/faqs.htm">FAQ selbst</a>
</body></html>
"""


def test_collect_info_links_filters_and_dedupes():
    links = collect_info_links(SAMPLE_HTML, "https://www.chrono24.de")
    assert links == [
        "https://www.chrono24.de/info/buyer-protection.htm",
        "https://www.chrono24.de/info/faqs.htm",
        "https://www.chrono24.de/info/trusted-checkout.htm",
    ]


def test_url_filename_roundtrip():
    url = "https://www.chrono24.de/info/buyer-protection.htm"
    name = url_to_filename(url)
    assert name == "info__buyer-protection.htm.html"
    assert filename_to_url(name) == url
