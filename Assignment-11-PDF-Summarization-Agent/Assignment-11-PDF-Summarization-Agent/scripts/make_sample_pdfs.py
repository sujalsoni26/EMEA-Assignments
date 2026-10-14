"""Generate the sample PDFs used for demos and tests.

    python scripts/make_sample_pdfs.py

Creates in ``samples/``:
  * solar_power_report.pdf   - a 3-page report with running headers/footers
  * invalid/password_protected.pdf - encrypted PDF (password: "secret")
  * invalid/scanned_image_only.pdf - a page that is only an image (no text layer)
  * invalid/not_really_a.pdf       - a text file with a .pdf extension
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"

CSS = """
body { font-family: sans-serif; font-size: 11pt; line-height: 1.45; }
h1 { font-size: 20pt; margin-bottom: 4pt; }
h2 { font-size: 13.5pt; margin-top: 14pt; margin-bottom: 4pt; }
p { margin-bottom: 7pt; text-align: justify; }
.subtitle { color: #555555; font-size: 11pt; margin-bottom: 14pt; }
"""

SOLAR_HTML = """
<h1>The Rise of Solar Power</h1>
<p class="subtitle">A short history of photovoltaic technology, its economics and its challenges</p>

<h2>1. Early history</h2>
<p>For most of the twentieth century, solar power was a curiosity. The photovoltaic effect, in which certain
materials produce an electric current when exposed to light, was first observed by the French physicist Edmond
Becquerel in 1839. More than a century later, in 1954, researchers at Bell Laboratories in the United States built
the first practical silicon solar cell. It converted about six percent of the sunlight that reached it into
electricity, which was impressive for the time but far too expensive for everyday use.</p>
<p>The first important customer for solar cells was the space industry. Satellites needed a reliable source of power
that did not run out, and cost mattered much less than weight and reliability. In 1958 the Vanguard 1 satellite
carried a small set of solar cells into orbit, and they continued to power its radio transmitter for several years.
For the next two decades, space programmes and remote installations such as lighthouses and telecommunications
towers were among the few places where solar electricity made economic sense.</p>
<p>The oil crises of the 1970s changed the political mood. When oil prices rose sharply, governments in Europe, the
United States and Japan began to fund research into alternative energy sources. Solar cells became more efficient and
slightly cheaper, but they remained far more expensive than electricity from coal, gas or nuclear power stations. In
the early 1980s, a solar module could cost more than twenty dollars for every watt of capacity, which meant that a
system for a single house would cost as much as a car.</p>

<h2>2. Policy and the collapse in prices</h2>
<p>The real turning point came in the 2000s, when a handful of countries introduced policies that guaranteed a fixed
price for electricity produced by solar panels. Germany's feed-in tariff, introduced in 2000, was the most
influential. It promised households and businesses a stable income for twenty years, and it created a large and
predictable market for solar panels. Manufacturers responded by building bigger factories, and the cost of
production began to fall quickly.</p>
<p>China played a decisive role in the next stage. Chinese companies invested heavily in manufacturing capacity for
polysilicon, wafers, cells and complete modules, supported by government loans and industrial policy. Between 2010
and 2020, the price of solar modules fell by roughly ninety percent. This collapse in prices was painful for many
manufacturers in Europe and the United States, several of which went bankrupt, but it made solar power affordable
for customers around the world.</p>
<p>Economists often describe this pattern using a learning curve. For solar modules, every doubling of the total
number of panels ever produced has been associated with a price reduction of around twenty percent. As the market
grew, engineers found ways to use thinner wafers, reduce the amount of silver in each cell and automate production
lines. Efficiency also improved, and typical commercial panels now convert more than twenty percent of sunlight into
electricity.</p>

<h2>3. Solar power today</h2>
<p>Today solar power is one of the cheapest ways to generate electricity in many parts of the world. In sunny
regions, large solar farms can produce electricity at a lower cost than new coal or gas plants. Solar has become the
fastest-growing source of new electricity capacity globally, and in several recent years more solar capacity was
installed than all other power sources combined. Rooftop systems have also become common in countries such as
Australia, where a large share of houses now have panels on their roofs.</p>

<h2>4. Challenges: variability and the grid</h2>
<p>The rapid growth of solar power creates new challenges. The sun does not shine at night, and output falls on
cloudy days and in winter. When many solar systems produce electricity at the same time, prices in the middle of the
day can drop sharply, sometimes even below zero, while demand in the evening must still be met by other sources.
Grid operators describe this pattern as the duck curve, because the graph of net demand over a day resembles the
shape of a duck.</p>
<p>Several solutions are being used to manage this variability. Battery storage is the most visible one. The price
of lithium-ion batteries has fallen dramatically thanks to the growth of electric vehicles, and large battery systems
can now store cheap solar electricity during the day and release it in the evening. Other approaches include stronger
connections between regional grids, so that electricity can be moved from places with surplus power to places with
shortages, and smart pricing that encourages consumers to use electricity when it is plentiful, for example by
charging electric cars at midday.</p>

<h2>5. Environmental and social concerns</h2>
<p>There are also concerns about the environmental and social impact of the solar industry itself. Manufacturing
panels requires energy and raw materials, and some supply chains have been linked to forced labour, which has led to
new import rules in several countries. Old panels will need to be recycled in large numbers in the coming decades,
and recycling methods that recover valuable materials such as silver and high-purity silicon are still being
developed. Large solar farms can also compete with agriculture and nature for land, although projects that combine
solar panels with grazing or crops are becoming more common.</p>

<h2>6. Outlook</h2>
<p>Despite these challenges, most energy forecasts expect solar power to keep growing rapidly. International agencies
project that solar could become the largest source of electricity in the world within the next two decades.
Researchers are working on new cell designs, such as tandem cells that stack perovskite materials on top of silicon to
capture more of the solar spectrum, and laboratory versions have already exceeded thirty percent efficiency. If these
technologies can be manufactured reliably and cheaply, they could push the cost of solar electricity even lower.</p>
<p>The story of solar power shows how public policy, industrial investment and steady engineering improvements can
transform an expensive niche technology into a mainstream source of energy. What began as a laboratory curiosity and a
power source for satellites is now a central part of the global effort to reduce greenhouse gas emissions and limit
climate change.</p>
"""


def html_to_pdf(html: str, path: Path, *, title: str, header: str) -> None:
    """Render HTML to an A4 PDF, then stamp a running header and a page-number footer."""
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (60, 72, -60, -72)
    story = pymupdf.Story(html=html, user_css=CSS)
    writer = pymupdf.DocumentWriter(str(path))
    more = True
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()

    doc = pymupdf.open(path)
    for page in doc:
        page.insert_text((60, 45), header, fontsize=8.5, color=(0.45, 0.45, 0.45))
        footer = f"Page {page.number + 1} of {doc.page_count}"
        page.insert_text((mediabox.width / 2 - 25, mediabox.height - 35), footer, fontsize=8.5, color=(0.45, 0.45, 0.45))
    doc.set_metadata({"title": title, "author": "EMEA AI Cohort 1 sample", "subject": "Sample document"})
    doc.save(path.with_suffix(".tmp.pdf"), garbage=3, deflate=True)
    doc.close()
    path.with_suffix(".tmp.pdf").replace(path)


def make_invalid_samples(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    # 1. Password-protected PDF
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "This document is confidential and protected by a password.", fontsize=12)
    doc.save(
        folder / "password_protected.pdf",
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="secret",
    )
    doc.close()

    # 2. "Scanned" PDF: render a text page to an image and embed only the image.
    source = pymupdf.open()
    page = source.new_page()
    page.insert_textbox(
        page.rect + (72, 72, -72, -72),
        "This page was scanned on a photocopier. It looks like text, but the PDF only contains a picture "
        "of the text, so there is no text layer that can be extracted without OCR. " * 4,
        fontsize=12,
    )
    image = page.get_pixmap(dpi=100, colorspace=pymupdf.csGRAY).tobytes("jpeg", jpg_quality=60)
    source.close()
    scanned = pymupdf.open()
    scanned_page = scanned.new_page()
    scanned_page.insert_image(scanned_page.rect, stream=image)
    scanned.save(folder / "scanned_image_only.pdf")
    scanned.close()

    # 3. Not a PDF at all
    (folder / "not_really_a.pdf").write_text("This is a plain text file that was renamed to .pdf.\n")


def main() -> None:
    SAMPLES.mkdir(exist_ok=True)
    html_to_pdf(
        SOLAR_HTML,
        SAMPLES / "solar_power_report.pdf",
        title="The Rise of Solar Power",
        header="Energy Insights Series  |  The Rise of Solar Power",
    )
    make_invalid_samples(SAMPLES / "invalid")
    for path in sorted(SAMPLES.rglob("*.pdf")):
        print(f"{path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
