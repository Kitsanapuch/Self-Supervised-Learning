# แผนการทดลองที่กำหนดก่อนดูผล

## คำถามวิจัย
RQ1: Masked-reconstruction SSL ช่วย Macro-F1 ในการจำแนกกิจกรรมบนผู้ใช้ที่ไม่เคยเห็น เมื่อ training labels มีน้อยหรือไม่?
RQ2: temporal, channel และ combined masking ให้ผลต่างกันอย่างไร?
RQ3: ความได้เปรียบดังกล่าวยังอยู่เมื่อปิดกลุ่ม accelerometer หรือ gyroscope หรือไม่?

## Protocol
- Dataset: UCI HAR official split, 6 channels, subject-disjoint validation split seed 2026
- Seeds: 0,1,2; label budgets: 1%,5%,10%,100% of training examples, rounded up per class
- Methods: supervised, SSL-time, SSL-channel, SSL-both
- 30 fine-tuning epochs; SSL 30 pretraining epochs; Adam lr=.001; batch=128
- Fixed architecture, same labeled examples and classifier head initialization across methods
- Early stopping ไม่ใช้; train ครบ budget แล้วเลือก checkpoint ที่ validation F1 สูงสุด
- ทุกวิธีใช้ validation labels เต็มชุด; test labels ใช้ประเมินหลังเลือก checkpoint
- Primary endpoint: clean test Macro-F1; secondary: missing accelerometer / missing gyroscope, accuracy and per-class F1
- Report mean and sample standard deviation across seeds; paired SSL-minus-supervised differences
- 3 seeds ไม่พอสำหรับข้อสรุป significance ที่แข็งแรง ไม่ทำการอ้าง p-value จาก window-level independence
- Fixed finetuning epochs เท่ากัน แต่ SSL มี compute เพิ่มในการ pretrain; ไม่อ้าง compute-matched comparison
- ห้ามเปลี่ยน hyperparameters ตาม test results; การปรับวิธีภายหลังต้องใช้ validation และแยก exploratory study

## ข้อจำกัดที่ต้องเปิดเผย
หนึ่ง dataset, หนึ่ง validation split, สาม seeds, windows ซ้อนทับภายใน subject, validation annotation budget แยกจาก training budget, derived benchmark labels ไม่ใช่การติดป้ายภาคสนาม, sensor failure จำลอง, single-channel training masking เทียบกับ group outage, mask coverage ไม่เท่ากัน, ไม่มี SOTA contrastive baselines, ไม่มีการปรับ hyperparameter budget อย่างเป็นระบบ

## ก่อนส่งบทความ
เพิ่ม dataset อิสระ, repeat subject splits, TS-TCC/TS2Vec baseline ที่ reproduce จริง, mask-rate-matched ablation, probe แบบ freeze encoder, เพิ่ม seeds/subject-level uncertainty, ทบทวน literature ล่าสุดและตรวจ novelty กับอาจารย์ การรันทดลองครบชุดนี้ไม่ใช่การรับรองว่าบทความจะได้รับตีพิมพ์
