# เริ่มอ่านตรงนี้

## ชุดงานนี้ทำอะไร

งานเปรียบเทียบการจำแนกกิจกรรมจากเซนเซอร์ ด้วย supervised learning และ self-supervised learning แบบปิดบังสัญญาณ 3 รูปแบบ โดยทดสอบจำนวนป้ายกำกับที่ต่างกันและการขาดหายของเซนเซอร์

นี่เป็นงานทดลองที่รันจริงครบขอบเขตที่ระบุ พร้อมต้นฉบับให้แก้ไขและให้อาจารย์ตรวจ ไม่ใช่การรับรองว่าวิทยานิพนธ์เสร็จหรือบทความได้รับตีพิมพ์แล้ว

## ไฟล์ที่ควรเปิดตามลำดับ

1. `results/research-report.pdf` — รายงานภาษาอังกฤษพร้อมผลจริง กราฟ และข้อจำกัด
2. `results/RESULTS_TH.md` — สรุปผลภาษาไทย
3. `docs/DEFENSE_TH.md` — สรุปสิ่งที่ทำ คำถามอาจารย์พร้อมแนวตอบ และปัญหาที่เจอ
4. `docs/PAPER_DRAFT.md` — ต้นฉบับบทความที่แก้ไขข้อความได้
5. `README.md` — วิธีติดตั้งและรันซ้ำ
6. `docs/PROTOCOL_TH.md` — เงื่อนไขการทดลอง
7. `docs/RELATED_WORK.md` — งานใกล้เคียงและประเด็นความใหม่
8. `results/all_evaluations.csv` — ผลทั้งหมด 144 แถว
9. `results/paired_differences.csv` — ผลต่างที่จับคู่ seed
10. `results/audit.json` — ผลตรวจความครบถ้วนและตัวชี้วัด
11. `START_COLAB.ipynb` — ทางเลือกสำหรับรันบน Google Colab (ยังไม่ได้ทดสอบในบริการ Colab จริง)

## อ่านคะแนนอย่างไร

Macro-F1 ให้น้ำหนักทั้ง 6 กิจกรรมเท่ากัน ค่าในตารางเป็นเปอร์เซ็นต์: ยิ่งสูงยิ่งดี เครื่องหมาย +/- คือ sample standard deviation จาก 3 seed ไม่ใช่ confidence interval และไม่รับประกันผลกับผู้ใช้ใหม่ทุกคน

Accuracy คือสัดส่วนตัวอย่างที่จำแนกถูก ส่วน Macro-F1 พิจารณาทั้ง precision และ recall รายคลาส จึงไม่ควรเรียกตัวเลข Macro-F1 ว่า accuracy

1% labels หมายถึงข้อมูลติดป้ายที่ใช้ฝึก ไม่รวมป้ายกำกับ validation ที่ยังใช้เต็มชุด ปริมาณจริงปัดขึ้นทีละคลาสและบันทึกใน metadata

## สิ่งที่มีให้แล้ว

- source code พร้อม Git history สำหรับเริ่มพัฒนาและเผยแพร่
- ข้อมูลอ้างอิงและสคริปต์โหลดข้อมูล; ไม่รวม raw dataset
- supervised baseline และ SSL-time / SSL-channel / SSL-combined
- การทดลอง 4 budgets × 4 methods × 3 seeds
- ทดสอบ clean / ไม่มี accelerometer / ไม่มี gyroscope
- model weights ของทุก run ในแพ็กเกจฉบับเต็ม
- environment lock, split IDs, label indices, learning curves, confusion matrices
- รายงานผลจริงและบทความฉบับให้อาจารย์ตรวจ

## ใช้โมเดลที่ฝึกแล้ว

เมื่อแตกแพ็กเกจฉบับเต็มและติดตั้ง dependencies ให้ใช้ข้อมูล `.npy` รูปร่าง `[N,6,128]` ในหน่วยและ preprocessing เดียวกับ UCI HAR ตามลำดับ total_acc_x/y/z แล้ว body_gyro_x/y/z:

```bash
python -m src.predict --run runs/full/both_f0.1_s0 --input your_windows.npy --output predictions.json
```

ตัวอย่างนี้เลือก run เพื่อแสดงคำสั่งเท่านั้น ไม่ใช่ประกาศว่าเป็นโมเดลที่ดีที่สุด โค้ดจะใช้ค่า normalization จาก training metadata ให้อัตโนมัติ ความน่าจะเป็นจาก softmax ยังไม่ได้ calibration

## ก่อนส่งให้อาจารย์หรือส่งตีพิมพ์

ใส่ชื่อผู้เขียนและสังกัดจริง เลือกวารสาร/งานประชุม ตรวจรูปแบบและนโยบาย AI ของสถาบัน ให้คุณอ่านและเข้าใจโค้ดกับผลจริงก่อนรับรองความถูกต้อง

แนว time-channel masking มีงานทำมาก่อน จึงไม่ควรใช้คำว่าเสนอวิธีใหม่โดยไม่มีหลักฐาน งานฉบับนี้ยังมีหนึ่ง dataset และไม่มีการ reproduce SOTA baseline: เหมาะเป็นต้นฉบับให้ที่ปรึกษาประเมินและกำหนด contribution ต่อ ไม่ควรส่งโดยอ้างว่าเป็นผลเหนือกว่างานล่าสุด

## GitHub

สร้าง repository ในเครื่องแล้ว แต่ยังไม่ได้เผยแพร่ไปยังบัญชี GitHub ใด source ZIP ไม่มีโฟลเดอร์ `.git` แต่มี `repository.bundle` สำหรับกู้ประวัติเดิม ดู [คู่มืออัปโหลด GitHub](docs/GITHUB_PUBLISH_TH.md) ซึ่งมีคำสั่งตั้งแต่แตก ZIP จน push ครบ

```bash
git clone sensor-ssl/repository.bundle sensor-ssl-public
cd sensor-ssl-public
```

เลือกชื่อ repo และ license ร่วมกับอาจารย์ก่อน publish ข้อมูล raw และ model weights ถูก ignore ตามปกติ หากจะแชร์ weights ให้ใช้ release asset แทน commit ไฟล์ใหญ่
