# RAG evaluation

Knowledge base: the PDFs in `samples/`. LLM: `llama3`, embeddings: `nomic-embed-text`, top-k = 4, similarity threshold = 0.55.

## Summary

| Metric | Result |
|---|---|
| Retrieval hit rate (answer page retrieved and judged relevant) | 16/16 |
| Answer accuracy (answered and mentions the expected facts) | 16/16 |
| Safe refusals for questions not covered by the documents | 6/6 |
| Top similarity, answerable questions (min / median / max) | 0.56 / 0.74 / 0.87 |
| Top similarity, unanswerable questions (min / median / max) | 0.51 / 0.62 / 0.73 |

## Details

| Type | Question | Top score | Retrieval hit | Status | Correct |
|---|---|---:|:---:|---|:---:|
| answerable | How many days of paid annual leave do full-time employees get? | 0.87 | yes | answered | yes |
| answerable | What are the core working hours at Northwind Labs? | 0.83 | yes | answered | yes |
| answerable | For how many days per year can I work remotely from another country? | 0.80 | yes | answered | yes |
| answerable | What is the maximum hotel cost per night in London? | 0.71 | yes | answered | yes |
| answerable | How much is the yearly learning budget? | 0.59 | yes | answered | yes |
| answerable | How long is the partner's paid parental leave? | 0.78 | yes | answered | yes |
| answerable | What is the notice period after two years of employment? | 0.70 | yes | answered | yes |
| answerable | How often should the RO membrane of the AquaPure X2 be replaced? | 0.75 | yes | answered | yes |
| answerable | What does error code E2 mean on the purifier? | 0.66 | yes | answered | yes |
| answerable | What does E2 mean and what should I do? | 0.58 | yes | answered | yes |
| answerable | What is the part number of the RO membrane? | 0.56 | yes | answered | yes |
| answerable | How do I reset the filter timer after replacing a filter? | 0.75 | yes | answered | yes |
| answerable | How long is the warranty if I register the purifier? | 0.73 | yes | answered | yes |
| answerable | What is the tank capacity of the AquaPure X2? | 0.74 | yes | answered | yes |
| answerable | What are the four functions of the AI RMF Core? | 0.85 | yes | answered | yes |
| answerable | Which characteristics make an AI system trustworthy according to the AI RMF? | 0.85 | yes | answered | yes |
| not in docs | What is the capital of Australia? | 0.52 | - | insufficient_context | yes |
| not in docs | Who won the 2022 FIFA World Cup? | 0.51 | - | insufficient_context | yes |
| not in docs | How much does the AquaPure X2 cost? | 0.72 | - | insufficient_context | yes |
| not in docs | What is the dress code at Northwind Labs? | 0.72 | - | insufficient_context | yes |
| not in docs | How do I bake sourdough bread? | 0.52 | - | insufficient_context | yes |
| not in docs | What was Northwind Labs' revenue in 2025? | 0.73 | - | insufficient_context | yes |

## Answers

**Q: How many days of paid annual leave do full-time employees get?**

> According to [1], full-time employees receive 25 days of paid annual leave per year.

**Q: What are the core working hours at Northwind Labs?**

> According to [1], the core working hours at Northwind Labs are from 10:00 to 15:00 (local time) for meetings and collaboration.

**Q: For how many days per year can I work remotely from another country?**

> According to [1], you can work remotely from another country for up to 20 working days per calendar year.

**Q: What is the maximum hotel cost per night in London?**

> According to [1], the maximum hotel cost per night in London is EUR 220.

**Q: How much is the yearly learning budget?**

> According to [1], each employee has a personal learning budget of EUR 1,500 per calendar year, which can be used for courses, certifications, books and conferences related to their role.

**Q: How long is the partner's paid parental leave?**

> According to [1], the partner of the birth parent, including adoptive parents and same-sex partners, receives 6 weeks of fully paid leave, which can be taken at any time during the first year after the birth or adoption.

**Q: What is the notice period after two years of employment?**

> According to [10], after the first two years of employment, the notice period is two months.

**Q: How often should the RO membrane of the AquaPure X2 be replaced?**

> According to [3], the RO membrane should be replaced every 24 months.

**Q: What does error code E2 mean on the purifier?**

> Error code E2 on the purifier means "Leak detected. Unplug the purifier, dry the leak sensor under the unit and check all hose connections." [1]

**Q: What does E2 mean and what should I do?**

> According to [2], Error E2 means "Leak detected". To resolve this issue, you should "Unplug the purifier, dry the leak sensor under the unit and check all hose connections".

**Q: What is the part number of the RO membrane?**

> According to [1], the part number of the RO membrane is AP-RO2.

**Q: How do I reset the filter timer after replacing a filter?**

> To reset the filter timer after replacing a filter, you need to press and hold the FILTER button for 5 seconds until the indicator turns blue [1].

**Q: How long is the warranty if I register the purifier?**

> According to [1], the warranty is extended to 5 years when you register the product within 60 days of purchase.

**Q: What is the tank capacity of the AquaPure X2?**

> According to [1], the tank capacity of the AquaPure X2 is 6 litres.

**Q: What are the four functions of the AI RMF Core?**

> The four functions of the AI RMF Core are: GOVERN, MAP, MEASURE, and MANAGE. [1]

**Q: Which characteristics make an AI system trustworthy according to the AI RMF?**

> According to the AI RMF, the characteristics that make an AI system trustworthy are:
> 
> * Valid and reliable [1]
> * Safe [1]
> * Secure and resilient [1]
> * Accountable and transparent [1]
> * Explainable and interpretable [1]
> * Privacy-enhanced [1]
> * Fair with harmful bias managed [1]
> 
> These characteristics are mentioned in [1] as the characteristics of trustworthy AI systems.

**Q: What is the capital of Australia?**

> I could not find enough information in the uploaded documents to answer this question. Try rephrasing it, or upload a document that covers this topic.

**Q: Who won the 2022 FIFA World Cup?**

> I could not find enough information in the uploaded documents to answer this question. Try rephrasing it, or upload a document that covers this topic.

**Q: How much does the AquaPure X2 cost?**

> I could not find enough information in the uploaded documents to answer this question. Try rephrasing it, or upload a document that covers this topic.

**Q: What is the dress code at Northwind Labs?**

> I could not find enough information in the uploaded documents to answer this question. Try rephrasing it, or upload a document that covers this topic.

**Q: How do I bake sourdough bread?**

> I could not find enough information in the uploaded documents to answer this question. Try rephrasing it, or upload a document that covers this topic.

**Q: What was Northwind Labs' revenue in 2025?**

> I could not find enough information in the uploaded documents to answer this question. Try rephrasing it, or upload a document that covers this topic.

