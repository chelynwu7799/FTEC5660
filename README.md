# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution: 
> to students: please fill your solution description here.

My design separates perception from computation: the vision model only
reads three raw numbers off each receipt, while all arithmetic is done
deterministically in Python. The chain built in build_chain() is
ChatPromptTemplate | ChatDeepSeek | JsonOutputParser, using the required
deepseek-v4-flash-vision-exp model at temperature 0. Each receipt image is
sent as a base64 data URL in a multimodal human message, and the prompt asks
for strict JSON with three fields: final_payment (the amount actually paid,
after the ROUNDING line), subtotal (after discounts, before rounding), and
discounts (every discount/promotion/coupon/member/percentage-off line summed
as one positive number, rounding excluded). In answer_queries() the chain is
fanned out with chain.batch(..., return_exceptions=True) so all receipts are
extracted in parallel; any receipt whose call or JSON parsing fails is retried
once and otherwise skipped with a warning, so the program always finishes and
always writes results.csv. Aggregation uses Decimal: Query 1 sums the
per-receipt final_payment, and Query 2 sums subtotal + discounts per
receipt (rounding is never added back). Keeping the math out of the model
avoids LLM arithmetic errors, and each final response is formatted as a single
HK$xxxx.xx amount so the automatic grader sees exactly one number. On the
public test set both queries are marked correct.


## Chain design

```mermaid
flowchart LR
    A[Receipt images&lt;br/&gt;in folder] --&gt; B[Extraction chain&lt;br/&gt;ChatPromptTemplate&lt;br/&gt;→ ChatDeepSeek&lt;br/&gt;deepseek-v4-flash-vision-exp, temp=0&lt;br/&gt;→ JsonOutputParser]
    B --&gt;|"batch() + return_exceptions&lt;br/&gt;one parallel call per receipt"| C[Per-receipt JSON:&lt;br/&gt;final_payment / subtotal / discounts]
    C --&gt;|failed receipts:&lt;br/&gt;one retry, else skip with warning| D[Deterministic aggregation&lt;br/&gt;in Python with Decimal]
    D --&gt; E["Q1 = Σ final_payment"]
    D --&gt; F["Q2 = Σ (subtotal + discounts)"]
    E --&gt; G["Single-amount answers:&lt;br/&gt;HK$xxxx.xx"]
    F --&gt; G
