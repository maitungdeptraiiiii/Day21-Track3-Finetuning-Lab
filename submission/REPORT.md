# Lab 21 — Evaluation Report

**Họ tên**: Mai Phan Anh Tùng  **MSSV**: 2A202602980  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Colab T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup — lựa chọn và lý do

| | |
|---|---|
| Base model | `unsloth/Qwen3.5-4B` (mặc định tier T4) |
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) — corpus mặc định |
| Train / val | 225 / 25 (seed 42) |
| Eval | 50 mẫu target + 15 câu regression (checksum cố định, không sửa) |
| `max_length` | 1024 (theo tier) — p95 đo được là **98** token, gợi ý **256** *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch → **30 step** (225 mẫu ÷ batch hiệu dụng 16 = 15 step/epoch), chung cho cả 4 run |
| Batch hiệu dụng | 1 × grad_accum 16 = **16** (< 32, deck §11.4) |
| Precision | fp16 (T4 không có bf16) |

**Vì sao chọn model + dataset này.** Qwen3.5-4B là model lớn nhất chạy bf16/fp16 LoRA vừa
16 GB của Colab T4 (~10 GB VRAM), có chat template hỗ trợ tiếng Việt và khối `<think>`.
Dataset mặc định được giữ vì mọi nhóm điểm đều có thang đo **khách quan** (so khớp từng
trường, JSON parse được, ms/mẫu) — không cần LLM judge — và checksum tập eval cố định giúp
phép so sánh với mốc đóng băng là công bằng. Cùng một base model được dùng cho mốc NB2
lẫn bản fine-tune NB3.

**Về `max_length`.** Phân bố độ dài (n=250): mean 93,1 · p50 93 · p95 98 · p99 100 · max
101 token. p95 gợi ý 256 (làm tròn lên luỹ thừa 2), tier đặt 1024. Vì mẫu dài nhất chỉ 101
token, **cả 256 lẫn 1024 đều không cắt mẫu nào** → tập huấn luyện sau tokenize giống hệt
nhau, kết quả không đổi. Tôi giữ 1024 để không lệch khỏi cấu hình tier mà `make verify` và
mã NB3/NB4 dùng; nếu dataset có câu trả lời dài hơn, tôi sẽ đặt theo p95.

**Template có giữ khối `<think>` không?** **Có** — *(results/template_check.json:
`open_tag_present=true`, `body_present=true`, verdict "reasoning preserved")*. Chuỗi render
với một trace mẫu vẫn giữ nguyên `<think>
buoc 1: ... buoc 2: ...
</think>`. Với corpus
mặc định, câu trả lời là JSON trần; template tự chèn `<think>

</think>` rỗng vào phần
generation prompt, nên `assistant-only`, `masked-think` và `response-only` cho cùng một mask.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39 / 94 token) — < 0.95 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss (giải mã ngược các vị trí `labels != -100`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn bị che (không tính loss):

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>

```

Đối chứng: với `MASK_MODE=everything`, supervised = 94/94 (100%) — toàn bộ system + câu hỏi
bị tính loss, đúng là lỗi kinh điển mà assert bắt được. `<|im_end|>` (EOS) nằm trong loss
nên model học được khi nào dừng.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train, đóng băng)

Nguồn: `results/baselines_frozen.json` (a, b) và `results/verdict.json` (c). Eval đầy đủ:
50 mẫu target, 15 câu regression, không dùng `EVAL_LIMIT`.

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.00 | 3263.7 |
| (b) base + optimized prompt + few-shot | 0.765 | 0.7911 | 1.00 | 1032.9 |
| (c) LoRA fine-tune (`correct`) | **0.970** | **0.5444** | 1.00 | 1399.0 |

**(b) có thật sự mạnh hơn (a) không?** Có, và chênh rất xa: 0.000 → 0.765. (a) được 0
điểm target vì format = 0. Không output nào của (a) parse được thành JSON đủ 4 khoá, nên
không trường nào được chấm đúng. Latency của (a) cũng gấp ~3 lần (b) (3264 so với 1033 ms).
Điều đó cho thấy với prompt đơn giản, model sinh văn bản dài (giải thích/suy luận) thay vì
JSON gọn. Prompt (b) có liệt kê nhãn hợp lệ và có few-shot nên giải quyết được cả format lẫn
độ dài. Regression của (a) và (b) giống hệt nhau (0.7911) vì cùng một model chưa train trả
lời cùng 15 câu phổ thông; prompt triage không áp vào nhóm này.

Tôi **không sửa** `OPTIMIZED_PROMPT`. `make verify` xác nhận SHA của prompt (b) không đổi.

---

## 4. Giải phẫu cấu hình sai (NB4)

Nguồn: `results/runs.csv` (train loss, tham số, VRAM, thời gian) và `results/autopsy.json`
(target NB5 §4). Cả 4 run cùng **30 step**, mỗi run đối chứng chỉ đổi **một** biến so với
`correct`:

| Run | biến bị đổi | vị trí | r | α | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | latency ms | train s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `correct` | — (chuẩn) | text-linear (12 module) | 16 | 32 | 32,464,896 | 1e-4 | 0.6255 | **0.970** | 1.00 | 1399.0 | 420.9 | 8.78 |
| `attn_only` | vị trí adapter | q,v (2 module) | 283 (matched) | 566 | 32,456,704 | 1e-4 | **0.5377** | **0.970** | 1.00 | 908.9 | 274.7 | 8.79 |
| `wrong_lr` | learning rate | text-linear | 16 | 32 | 32,464,896 | **1e-5** | 1.5702 | **0.000** | 0.00 | 5317.1 | 408.5 | 8.78 |
| `qlora` | 4-bit base | text-linear | 16 | 32 | 32,464,896 | 1e-4 | 0.7058 | **0.940** | 1.00 | 1801.8 | 477.3 | **3.86** |

Ngân sách tham số của `attn_only` lệch so với `correct` là |32,456,704 − 32,464,896| /
32,464,896 = **0.025%** (< 5%, `make verify`: "attn_only is a FAIR contrast"). Như vậy so
sánh này là so **vị trí**, không phải so **ngân sách**.

Thứ tự theo train loss: `attn_only` (0.538) < `correct` (0.626) < `qlora` (0.706) <
`wrong_lr` (1.570).
Thứ tự theo **target NB5** (thước đo dùng để xếp hạng): `correct` = `attn_only` (0.970) >
`qlora` (0.940) > `wrong_lr` (0.000).

Hai thứ tự **không trùng nhau**. Theo loss, `attn_only` là run tốt nhất, thấp hơn `correct`
0.088. Theo target, nó chỉ **hoà** `correct`: loss thấp hơn không mua thêm được trường đúng
nào. Nếu xếp hạng bằng `final_loss`, tôi sẽ kết luận sai rằng "attention-only tốt hơn
all-linear".

**4.1 — Vị trí vs rank.** `attn_only` dồn cùng ngân sách ~32,46M tham số vào 2 module (q,v)
với r = 283, còn `correct` rải ra 12 module với r = 16. Trên tập target hai run **hoà nhau
hoàn toàn**: 0.970 so với 0.970, format đều 1.00. Bằng chứng này nói rằng **với bài này, ở
ngân sách này, cả vị trí lẫn rank đều không phải đòn bẩy**. Nhiệm vụ (ánh xạ một ticket ngắn
sang 4 nhãn đóng) đủ đơn giản để 32M tham số đặt ở đâu cũng học được. Kết quả **không** bác
bỏ được khuyến nghị all-linear của deck §11.2, vì phép thử đã bão hoà: cả hai đều ở 0.97 và lỗi
còn lại là cùng một loại (xem mục 6), nên không còn chỗ để thấy khác biệt. Muốn phân biệt được
thì cần bài khó hơn hoặc ngân sách nhỏ hơn nhiều (ví dụ quét r ∈ {1, 4}).
Khác biệt đo được lại nằm ở **latency**: `attn_only` chạy 908.9 ms/mẫu so với 1399.0 ms của
`correct` (−35%), vì adapter chưa merge phải tính nhánh LoRA ở 2 thay vì 12 loại module mỗi
lớp. Train cũng nhanh hơn (274.7 so với 420.9 s). Train loss thấp hơn (0.538 so với 0.626) có
thể do rank lớn trên q,v khớp mẫu train nhanh hơn, nhưng không chuyển thành điểm target. Đây
chính là Lỗi #3: chấm bằng chỉ số thay thế.

**4.2 — `wrong_lr`.** Chỉ đổi LR từ 1e-4 xuống 1e-5 (thang full-FT), loss cuối cao gấp
~2,5 lần (1.570 so với 0.626) sau cùng 30 step. Với LoRA, B được khởi tạo bằng 0 và chỉ có
khoảng 0,8% tham số được cập nhật. LR thang full-FT quá nhỏ nên trong ngân sách 30 step
adapter gần như chưa rời điểm xuất phát. Nếu chỉ nhìn đường loss mà không biết LR, dễ kết
luận sai rằng "LoRA không học được bài này" hoặc "cần thêm epoch/rank", trong khi đòn bẩy
thật chỉ là nhân LR lên ×10.
NB5 xác nhận, và cho thấy mức độ nặng hơn loss gợi ý: `wrong_lr` được **target 0.000, format
0.00**, latency 5317.1 ms/mẫu. Đây gần đúng hồ sơ của baseline (a) (0.000 / 0.00 / 3263.7 ms):
model vẫn sinh văn bản dài không phải JSON, tức adapter gần như chưa đổi hành vi của base.
Loss 1.57 trông như "đang học dở", nhưng trên eval là **không học được gì**. Trong cả lab, đây
là biến có tác động lớn nhất: một con số LR quyết định giữa 0.970 và 0.000.

**4.3 — `qlora`.** Peak VRAM giảm từ 8.78 xuống 3.86 GB (**−56%**, tiết kiệm 4.92 GB).
Đổi lại, train chậm hơn 13% (477 so với 421 s, do phải dequantize mỗi lần forward) và loss
cuối cao hơn (0.706 so với 0.626). Trên eval: target **0.940** so với 0.970 (−0.030, tức 6
trường trên 200), format vẫn 1.00, latency **1801.8 ms** so với 1399.0 ms (+29%, vì base 4-bit
phải dequantize khi suy luận). Số đo **ủng hộ có điều kiện** khuyến nghị "không dùng QLoRA cho
Qwen3.5": khi đã vừa VRAM ở 16-bit (8.78 GB trên T4 16 GB) thì QLoRA chỉ làm mất điểm và chậm
hơn cả lúc train lẫn lúc suy luận. Nhưng mức mất 0.03 là nhỏ. Nếu phần cứng chỉ có ~6 GB (như
GPU laptop RTX 3060 của tôi), QLoRA là cách duy nhất để train được, và cái giá 3 điểm là chấp
nhận được. Với n = 50, chênh 0.03 nằm trong vùng nhiễu, nên tôi không kết luận mạnh hơn
"QLoRA không tốt hơn, và có thể kém hơn một chút".

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = −0.247` · `valid_trace_rate = 0.00`
*(results/verdict.json — Δ tính so với baseline (b))*

Lý do theo `verdict.json`: *"general capability regressed by 0.247 (tolerance 0.020)"*.

**Diễn giải.** Fine-tune thắng rõ ràng trên chính bài toán. Target tăng từ 0.765 lên 0.970
(+20,5 điểm %), tức sai từ ~23,5% số trường xuống còn 3%; format giữ nguyên 1.00. Nhưng bản
fine-tune **làm hỏng năng lực phổ thông**: regression rơi từ 0.791 xuống 0.544 (−24,7 điểm
%, gấp hơn 12 lần dung sai 0.02). Đây là quên thảm hoạ (catastrophic forgetting, deck §6.3).
225 mẫu train đều cùng một dạng (ticket → JSON 4 khoá, câu trả lời trần), không có mẫu phổ
thông nào. Với LR 1e-4 và LoRA gắn vào toàn bộ 12 loại linear của decoder, 30 step đủ để đẩy
model về "chế độ triage": gặp câu hỏi gì nó cũng có xu hướng trả lời theo khuôn ngắn/JSON.
Chính vị trí "all-linear" giúp target cao cũng là thứ cho adapter nhiều chỗ để ghi đè hành vi
chung. Latency cũng xấu đi +35% (1033 → 1399 ms) vì adapter chưa merge, mỗi lớp phải tính
thêm nhánh LoRA.

`valid_trace_rate = 0.00` là kỳ vọng chứ không phải lỗi: corpus mặc định là JSON trần, không
có trace suy luận, nên không có `<think>` nào để giữ.

Kết luận: nếu sản phẩm **chỉ** làm triage ticket thì bản fine-tune đáng giá (+20,5 điểm
target). Nhưng theo định nghĩa của lab (model phải giữ được năng lực chung), nó **chưa được
deploy**. Không nới cổng; cách sửa đúng là trộn 1–5% dữ liệu phổ thông (replay), giảm
epoch/LR, hoặc chỉ route ticket CSKH qua adapter (hot-swap) còn câu hỏi khác đi base model.

---

## 6. Định tính — bắt buộc có cả ca THUA

Nguồn: `results/qualitative.json` (dự đoán của (c), cắt ~100 ký tự) đối chiếu nhãn trong
`data/eval_target.jsonl`. **Hạn chế:** pipeline không lưu output từng mẫu của (b), chỉ lưu
điểm tổng 0.765. Vì vậy bảng dưới không có cột (b), và "FT thua" được định nghĩa là **FT trả
sai so với nhãn**. Ca thua lớn nhất của FT so với (b) nằm ở nhóm **regression** (0.791 → 0.544,
mục 5), nhưng output từng câu regression cũng không được lưu.

| # | Ticket (i) | Nhãn đúng | (c) fine-tune | điểm | Nhận xét |
|---|---|---|---|---|---|
| 1 | (#4) "...đèn bàn LED... Vỡ khi nhận. **Gấp.**" | san_pham_loi · **cao** · đèn bàn LED | san_pham_loi · cao · đèn bàn LED · … | 1.00 | ✅ Bắt đúng tín hiệu khẩn "Gấp" → cao |
| 2 | (#8) "...chuột không dây... **Bảo hành bao lâu.** Không vội." | hoi_thong_tin · thap · chuột không dây | hoi_thong_tin · thap · chuột không dây · … | 1.00 | ✅ Phân biệt hỏi thông tin với khiếu nại; trích đúng tên sản phẩm |
| 3 | (#30) "...đèn bàn LED... **Hoàn lại.** Sớm nhé. Lần cuối mua ở đây." | **doi_tra** · trung_binh · đèn bàn LED · tieu_cuc | doi_tra · trung_binh · đèn bàn LED · … | 1.00 | ✅ "Hoàn lại" (trả hàng) không bị nhầm với `hoan_tien` dù cùng chữ "hoàn" |
| 4 | (#3) "...bình giữ nhiệt... Chưa thấy tiền. **Khi nào tiện.** Cảm ơn shop nhiều." | hoan_tien · **thap** · bình giữ nhiệt · tich_cuc | hoan_tien · **trung_binh** · bình giữ nhiệt · … | 0.75 | ❌ **FT thua**: "Khi nào tiện" (lúc nào rảnh) phải là `thap` |
| 5 | (#39) "...nồi chiên không dầu... Hoàn tiền. **Khi nào tiện.** Quá tệ." | hoan_tien · **thap** · nồi chiên không dầu · tieu_cuc | hoan_tien · **trung_binh** · nồi chiên không dầu · … | 0.75 | ❌ **FT thua**: cùng lỗi urgency, dù sentiment tiêu cực |
| 6 | (#41) "...đèn bàn LED... Giao hàng chậm. **Khi nào tiện.** Cảm ơn shop nhiều." | van_chuyen · **thap** · đèn bàn LED · tich_cuc | van_chuyen · **trung_binh** · đèn bàn LED · … | 0.75 | ❌ **FT thua**: cùng lỗi urgency |

**Có mẫu chung ở các ca thua không? Có, và rất rõ.** Cả 6/50 ticket FT bị trừ điểm (#3, #5,
#12, #39, #41, #46) đều sai đúng **một trường, `urgency`**, và cả 6 đều chứa cụm **"Khi nào
tiện"**. Nhãn là `thap`, FT đoán `trung_binh`. Eval chỉ có đúng 6 ticket chứa cụm này, nên FT
sai **6/6** ca đó và không sai ở đâu khác: 6 × 0,25 = 1,5 trường trên 50 mẫu, đúng bằng
1 − 0.970.

Điều đáng chú ý: tập train (`data/split/train.jsonl`) có **30 mẫu** chứa "Khi nào tiện", cả 30
đều gán `thap`, nên lỗi **không** do thiếu dữ liệu. Giả thuyết của tôi: "Khi nào…" trong tiếng
Việt thường mở đầu câu hỏi về thời gian, nên prior của base model kéo về "đang hỏi tiến độ →
trung bình". 30 step với LR 1e-4 chưa đủ ghi đè prior đó cho một cụm cụ thể, trong khi các tín
hiệu rõ hơn ("Gấp", "Không vội") đã học được. Muốn kiểm chứng giả thuyết cần lưu output của (b)
trên 6 ca này, hoặc train thêm epoch rồi xem 6 ca có được sửa không.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Câu hỏi của lab là "fine-tune có thắng base model đã được prompt tử tế không".
Câu trả lời của tôi: **thắng trên bài toán, thua trên tổng thể, và không nên deploy ở dạng
này.** Trên target, LoRA đưa độ chính xác từ 0.765 (prompt tối ưu + few-shot) lên 0.970, và 6
lỗi còn lại cùng một nguyên nhân chỉ ra được (cụm "Khi nào tiện"). Nhưng cổng hồi quy FAILED vì
năng lực phổ thông rơi 0.247, gấp hơn 12 lần dung sai. Nguyên nhân nhân quả nằm ở dữ liệu: 225
mẫu cùng một khuôn và không có replay phổ thông. Thêm vào đó, 30 step với LR 1e-4 trên toàn bộ
linear đủ để kéo hành vi chung của model về "chế độ triage". Latency cũng tệ hơn (+35% khi chưa
merge).

Đâu là đòn bẩy thật? Theo bằng chứng của tôi, đó là **learning rate**: đổi đúng một con số LR
làm target đi từ 0.970 xuống 0.000. Ngược lại, **vị trí và rank không phải đòn bẩy ở bài này**:
cùng ngân sách tham số, q,v @ r=283 hoà all-linear @ r=16 (0.970 = 0.970), dù train loss khác
nhau 0.09. **Mask** là điều kiện tiên quyết: nó đúng (41% token được giám sát, câu hỏi bị che),
nên mọi so sánh phía sau mới có nghĩa. **QLoRA** tiết kiệm 56% VRAM với cái giá nhỏ (−0.03
target, +29% latency).

Nếu phải đưa vào sản phẩm, tôi sẽ không nới cổng mà làm một trong hai việc. Cách thứ nhất là
train lại với 1–5% dữ liệu phổ thông trộn vào (deck §6.3) rồi đo lại regression. Cách thứ hai
là giữ base model cho câu hỏi chung và chỉ hot-swap adapter khi request là ticket CSKH. Với cách
thứ hai, `attn_only` là ứng viên tốt hơn `correct`: cùng điểm target mà nhanh hơn 35%.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Train loss không phải thước đo để chọn run.** Trước lab, tôi mặc định run nào loss thấp
   hơn thì tốt hơn. Ở NB4, `attn_only` có loss thấp nhất (0.538, thấp hơn `correct` 0.088),
   và nếu dừng ở đó tôi đã chọn nó và viết rằng "attention-only tốt hơn all-linear". NB5 cho
   thấy hai run hoà đúng 0.970. Loss chỉ đo mức khớp với 225 mẫu train; nó không biết JSON có
   parse được hay nhãn có đúng không. Từ giờ, quyết định chọn cấu hình của tôi phải dựa trên
   eval của đúng bài toán, còn loss chỉ dùng để phát hiện run hỏng.

2. **LoRA nhạy với learning rate hơn tôi tưởng rất nhiều, và loss che mức độ hỏng.** Tôi từng
   nghĩ LR 1e-5 chỉ làm học chậm hơn một chút. Thực tế, cùng 30 step, `wrong_lr` ra target
   0.000 và format 0.00, gần như y hệt base model với prompt đơn giản (baseline (a)), tức là
   adapter không học được gì dùng được. Vậy mà loss 1.57 lại trông giống "đang học dở". Bài học
   cụ thể: khi chuyển từ full fine-tune sang LoRA phải nhân LR lên ~10×, và một run có loss
   "giảm chậm" phải được eval trước khi kết luận là cần thêm epoch hay thêm rank.

3. **Điểm tổng 0.97 che một lỗi có hệ thống, và tôi chỉ thấy khi đọc từng mẫu.** Nhìn con số
   0.97, tôi nghĩ lỗi còn lại là nhiễu rải rác. Khi đối chiếu `qualitative.json` với nhãn, cả
   6 ca sai đều là trường `urgency` của ticket chứa "Khi nào tiện" (sai 6/6), dù tập train có
   30 mẫu như vậy gán `thap`. Một lỗi lặp 100% trên một cụm từ là thứ người dùng thật sẽ gặp
   lại liên tục. Ngoài ra, cổng regression FAILED (0.791 → 0.544) cho tôi thấy "thắng trên bài
   toán" chưa phải "nên deploy": nếu chỉ đo target, tôi đã báo cáo một chiến thắng +20,5 điểm
   mà không biết mình vừa làm model kém đi ở mọi việc khác.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** (1) trộn ~5% câu hỏi phổ thông vào train để xem có qua
được cổng regression mà vẫn giữ target ≥ 0.95 không; (2) lưu output từng mẫu của (b) để so (b)
với (c) trên đúng 6 ca "Khi nào tiện"; (3) quét r ∈ {1, 4, 16} ở all-linear (B4) để tìm ngân
sách nhỏ nhất mà bài này vẫn bão hoà.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
