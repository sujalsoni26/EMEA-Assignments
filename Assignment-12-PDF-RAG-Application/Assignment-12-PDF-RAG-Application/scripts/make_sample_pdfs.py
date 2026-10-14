"""Generate the sample knowledge-base PDFs.

    python scripts/make_sample_pdfs.py

The two documents describe a *fictional* company (Northwind Labs) and product
(AquaPure X2). Because the facts are invented, the LLM cannot answer questions
about them from its own training data, which makes it easy to verify that the
answers really come from the retrieved passages.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"

CSS = """
body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.45; }
h1 { font-size: 19pt; margin-bottom: 2pt; }
h2 { font-size: 13pt; margin-top: 14pt; margin-bottom: 4pt; }
p, li { margin-bottom: 5pt; text-align: justify; }
.subtitle { color: #555555; margin-bottom: 12pt; }
table { border-collapse: collapse; margin: 6pt 0 10pt 0; }
td, th { border: 1px solid #999999; padding: 3pt 6pt; font-size: 10pt; }
th { background-color: #eeeeee; }
"""

HANDBOOK = """
<h1>Northwind Labs Employee Handbook</h1>
<p class="subtitle">Version 2026.1 &mdash; applies to all employees in the Netherlands, Portugal and Poland</p>

<h2>1. Welcome to Northwind Labs</h2>
<p>Northwind Labs was founded in 2014 in Rotterdam and designs smart water-treatment products for homes and small
businesses. Today we employ 340 people across three offices: Rotterdam (headquarters), Lisbon and Krak&oacute;w.
This handbook explains the policies that apply to everyone at Northwind Labs. If anything in this handbook is
unclear, please contact the People Team at people@northwind-labs.example.</p>

<h2>2. Working hours</h2>
<p>A full-time working week at Northwind Labs is 38 hours. We use flexible working hours: you may start your day at
any time between 07:30 and 10:00, but everyone is expected to be available during the core hours from 10:00 to 15:00
(local time) for meetings and collaboration. Overtime must be agreed with your manager in advance and is compensated
with time off in lieu at a rate of one hour off for every hour worked.</p>

<h2>3. Hybrid and remote work</h2>
<p>Northwind Labs follows a hybrid model. Employees work from the office at least two days per week. Tuesday and
Thursday are team days, when all members of a team are expected in the office. The other days may be worked from home.
</p>
<p>Employees may work remotely from another country for up to 20 working days per calendar year. Remote work abroad
requires approval from your manager and must be reported to the People Team at least two weeks before you travel, so
that tax and insurance rules can be checked. Remote work from outside the European Union is not permitted.</p>
<p>Every employee receives a one-off home office allowance of EUR 400 to buy a desk chair, monitor or other equipment
for working from home.</p>

<h2>4. Annual leave</h2>
<p>Full-time employees receive 25 days of paid annual leave per year. Part-time employees receive leave in proportion
to their contracted hours. After every five years of service you receive one extra day of leave per year, up to a
maximum of 30 days. Leave requests should be submitted in the HR portal at least two weeks in advance for periods
longer than three days.</p>
<p>Up to 5 unused leave days may be carried over to the next calendar year. Carried-over days must be used before
31 March; after that date they expire.</p>

<h2>5. Sick leave</h2>
<p>If you are ill, notify your manager before 09:30 on the first day of absence. A doctor's note is required when you
are absent for more than three consecutive working days. When you return to work, please register your recovery in
the HR portal.</p>

<h2>6. Parental leave</h2>
<p>The birth parent receives 16 weeks of fully paid parental leave. The partner of the birth parent, including
adoptive parents and same-sex partners, receives 6 weeks of fully paid leave, which can be taken at any time during
the first year after the birth or adoption. Additional unpaid parental leave is available according to local law.</p>

<h2>7. Learning and development</h2>
<p>Each employee has a personal learning budget of EUR 1,500 per calendar year, which can be used for courses,
certifications, books and conferences related to their role. In addition, every employee is entitled to two paid
learning days per year. Conference attendance requires approval from your manager, and travel for conferences is paid
from the learning budget.</p>

<h2>8. Business travel and expenses</h2>
<p>Expenses must be submitted in the Ledgerly expense app within 30 days of the purchase. A receipt is required for
every expense above EUR 25. The following limits apply to business travel:</p>
<table>
<tr><th>Expense</th><th>Limit</th></tr>
<tr><td>Hotel (standard)</td><td>EUR 160 per night</td></tr>
<tr><td>Hotel in London or Paris</td><td>EUR 220 per night</td></tr>
<tr><td>Meals while travelling</td><td>EUR 45 per day</td></tr>
<tr><td>Flights under 6 hours</td><td>Economy class</td></tr>
<tr><td>Flights of 6 hours or longer</td><td>Premium economy</td></tr>
</table>
<p>Train travel is preferred over flying for journeys of less than five hours.</p>

<h2>9. Equipment and IT security</h2>
<p>Every employee receives a laptop (MacBook Pro or Lenovo ThinkPad), which is replaced every three years. Multi-factor
authentication is mandatory for all company accounts, and a 1Password account is provided to store passwords. Lock
your screen whenever you leave your device. Suspicious emails must be reported to security@northwind-labs.example
without clicking any links. Lost or stolen devices must be reported to the IT service desk within 24 hours.</p>

<h2>10. Probation and notice periods</h2>
<p>New employees have a probation period of two months. After probation, the notice period is one month during the
first two years of employment and two months after that.</p>

<h2>11. Code of conduct and reporting concerns</h2>
<p>We expect everyone to treat colleagues, customers and partners with respect. Discrimination, harassment and
bullying are not tolerated. If you experience or witness behaviour that breaks this code, speak to your manager or the
People Team. You can also report concerns anonymously through the SafeVoice portal, which is managed by an independent
provider. Northwind Labs does not allow retaliation against anyone who raises a concern in good faith.</p>
"""

MANUAL = """
<h1>AquaPure X2 Water Purifier</h1>
<p class="subtitle">User manual &mdash; model AP-X2 &mdash; Northwind Labs</p>

<h2>1. Safety instructions</h2>
<p>Read this manual before installing the AquaPure X2. Connect the purifier only to the cold-water supply: the inlet
water temperature must stay between 5&deg;C and 38&deg;C. Do not install the unit outdoors or where it can freeze.
Unplug the purifier before cleaning it or replacing filters. Children should not play with the appliance.</p>

<h2>2. Technical specifications</h2>
<table>
<tr><th>Specification</th><th>Value</th></tr>
<tr><td>Filtration stages</td><td>4: sediment filter, activated carbon filter, reverse osmosis (RO) membrane, UV-C LED</td></tr>
<tr><td>Tank capacity</td><td>6 litres</td></tr>
<tr><td>Flow rate</td><td>1.2 litres per minute</td></tr>
<tr><td>Inlet water pressure</td><td>1.0 to 4.0 bar</td></tr>
<tr><td>Power consumption</td><td>48 W</td></tr>
<tr><td>Dimensions (W x D x H)</td><td>42 x 18 x 45 cm</td></tr>
<tr><td>Weight (empty)</td><td>7.8 kg</td></tr>
</table>

<h2>3. Installation</h2>
<p>1. Close the cold-water supply valve under the sink. 2. Connect the inlet hose to the supply valve using the
supplied adapter. 3. Connect the drain hose to the sink drain. 4. Plug in the power adapter and open the supply valve.
5. Before first use, let the purifier flush for 10 minutes and discard this water. The blue indicator light shows that
the purifier is ready.</p>

<h2>4. Indicator lights</h2>
<table>
<tr><th>Light</th><th>Meaning</th></tr>
<tr><td>Blue</td><td>Normal operation</td></tr>
<tr><td>Amber, blinking</td><td>A filter must be replaced within the next 2 weeks</td></tr>
<tr><td>Red</td><td>A filter has expired; replace it now</td></tr>
<tr><td>Red, blinking with a beep</td><td>Leak detected; the inlet valve closes automatically</td></tr>
</table>

<h2>5. Filter replacement</h2>
<p>Replace the filters according to the following schedule. Intervals are based on average household use of 10 litres
per day; replace filters earlier if your water is very hard or contains a lot of sediment.</p>
<table>
<tr><th>Component</th><th>Part number</th><th>Replace every</th></tr>
<tr><td>Sediment filter</td><td>AP-SF2</td><td>6 months</td></tr>
<tr><td>Activated carbon filter</td><td>AP-CF2</td><td>6 months</td></tr>
<tr><td>RO membrane</td><td>AP-RO2</td><td>24 months</td></tr>
<tr><td>UV-C LED module</td><td>AP-UV2</td><td>3 years</td></tr>
</table>
<p>To replace a filter, unplug the purifier, open the front cover, turn the filter a quarter turn anticlockwise and
pull it out. Insert the new filter and turn it clockwise until it clicks. After replacing a filter, reset the filter
timer by pressing and holding the FILTER button for 5 seconds until the indicator turns blue.</p>

<h2>6. Cleaning and maintenance</h2>
<p>Wipe the outside of the purifier with a soft, damp cloth. Do not use abrasive cleaners or solvents. Sanitise the
water tank every 6 months with the AquaPure cleaning kit (part AP-CK1). If the purifier will not be used for more than
two weeks, close the supply valve and empty the tank; flush the system for 5 minutes before using it again.</p>

<h2>7. Troubleshooting</h2>
<table>
<tr><th>Problem or code</th><th>Possible cause and solution</th></tr>
<tr><td>Low water flow</td><td>Check that the supply valve is fully open. Replace the sediment filter if it is older than 6 months.</td></tr>
<tr><td>Water tastes or smells unusual</td><td>Flush the system for 5 minutes. If the problem continues, replace the activated carbon filter.</td></tr>
<tr><td>Error E1</td><td>Inlet water pressure is too low (below 1.0 bar). Check the supply valve and household water pressure.</td></tr>
<tr><td>Error E2</td><td>Leak detected. Unplug the purifier, dry the leak sensor under the unit and check all hose connections.</td></tr>
<tr><td>Error E3</td><td>UV-C module fault. Do not drink the water; contact customer support.</td></tr>
</table>

<h2>8. Warranty and support</h2>
<p>The AquaPure X2 comes with a 2-year standard warranty. The warranty is extended to 5 years when you register the
product within 60 days of purchase at northwind-labs.example/register. Filters and the cleaning kit are consumables and
are not covered by the warranty. Damage caused by incorrect installation, freezing or connection to hot water is not
covered.</p>
<p>Customer support is available by email at support@northwind-labs.example and by phone on +31 10 555 0199, Monday
to Friday from 08:30 to 17:30 (CET).</p>
"""


def html_to_pdf(html: str, path: Path, *, title: str, header: str) -> None:
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (60, 70, -60, -70)
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
        page.insert_text((mediabox.width / 2 - 25, mediabox.height - 35), f"Page {page.number + 1} of {doc.page_count}",
                         fontsize=8.5, color=(0.45, 0.45, 0.45))
    doc.set_metadata({"title": title, "author": "Northwind Labs (fictional sample)"})
    tmp = path.with_suffix(".tmp.pdf")
    doc.save(tmp, garbage=3, deflate=True)
    doc.close()
    tmp.replace(path)


def main() -> None:
    SAMPLES.mkdir(exist_ok=True)
    html_to_pdf(HANDBOOK, SAMPLES / "northwind_employee_handbook.pdf",
                title="Northwind Labs Employee Handbook", header="Northwind Labs  |  Employee Handbook 2026.1")
    html_to_pdf(MANUAL, SAMPLES / "aquapure_x2_user_manual.pdf",
                title="AquaPure X2 User Manual", header="Northwind Labs  |  AquaPure X2 User Manual")
    for path in sorted(SAMPLES.glob("*.pdf")):
        print(f"{path.relative_to(ROOT)}  ({pymupdf.open(path).page_count} pages)")


if __name__ == "__main__":
    main()
