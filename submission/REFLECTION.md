# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**Họ tên:** Mai Phan Anh Tùng · **MSSV:** 2A202602980

**1. Điều gì làm bạn ngạc nhiên nhất?**

Run `wrong_lr` chỉ đổi đúng một con số (LR 1e-4 → 1e-5), loss cuối 1.570 trông như "đang học
dở", vậy mà trên eval nó được target 0.000 và format 0.00, gần như y hệt base model với prompt
đơn giản. Tôi cũng ngạc nhiên khi `attn_only` có loss thấp nhất (0.538) lại chỉ hoà `correct`
(0.970 = 0.970). Hai kết quả này cùng nói một điều: loss và điểm thật có thể đi hai hướng.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Phần lớn thời gian là chờ pipeline NB1→NB5 chạy trên Colab T4 (khoảng 2 tiếng), đúng như dự
đoán. Thứ tôi không dự đoán là những việc "hậu cần": suýt nộp bằng chế độ rút gọn vì ô Colab
để sẵn `EVAL_LIMIT = 8`; loay hoay tải `out.zip` từ Colab về máy; và `verify` báo tập eval bị
sửa dù tôi không đụng vào. Lỗi cuối cùng hoá ra do Git trên Windows (`core.autocrlf=true`) đổi
ký tự xuống dòng LF → CRLF, làm checksum lệch đúng 50 byte cho 50 dòng.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng tin "fine-tune thì chắc chắn tốt hơn prompt" và "loss thấp hơn là model tốt hơn".
Kết quả của tôi bác cả hai. Prompt tối ưu + few-shot đã đạt 0.765 mà không cần train; fine-tune
lên được 0.970 nhưng làm regression rơi từ 0.791 xuống 0.544, nên phán quyết là FAILED. Thắng
trên bài toán chưa có nghĩa là nên deploy.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng Claude Code để hướng dẫn các bước chạy trên Colab, chạy NB1 và `verify` trên máy, đối
chiếu dự đoán với nhãn, và soạn report. Những chỗ nó sai hoặc thiếu mà tôi phải sửa lại:
- Nó báo file zip nộp bài đã nằm trên Desktop, nhưng thực ra file nằm trong thư mục repo; sau
  đó nó tự phát hiện và chuyển ra đúng chỗ.
- Bản report đầu tiên có 3 con số không lấy từ `results/` ("~10 GB VRAM" lấy từ README, "khoảng
  0,8% tham số" là ước lượng, "~6 GB" là đoán). Chỉ khi tôi hỏi lại "mọi con số có khớp
  `results/` không", nó mới chạy script đối chiếu và thay bằng số đo thật.
- Vì `qualitative.json` không lưu output của (b), các "ca thua" trong report là ca fine-tune
  sai so với nhãn, chưa phải so trực tiếp (b) với (c) trên từng mẫu.
Bài học: AI soạn nhanh nhưng phải bắt nó kiểm chứng từng con số với file gốc.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Đóng băng tập eval và đo baseline prompt tốt nhất trước khi train, gồm cả một bộ câu hỏi
regression cho các việc model đang làm tốt. Nếu prompt đã đủ dùng thì không fine-tune. Nếu vẫn
fine-tune, tôi trộn sẵn 1–5% dữ liệu phổ thông vào tập train, vì lab này cho thấy thiếu replay
có thể làm mất 0.247 điểm regression chỉ sau 30 step.
