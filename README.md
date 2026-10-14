# Sensor SSL: reproducible master's research project

โปรเจกต์วิจัย Self-Supervised Learning สำหรับจำแนกกิจกรรมจากเซนเซอร์ เมื่อข้อมูลติดป้ายกำกับมีน้อย พร้อมการทดลองสัญญาณขาดหาย สถานะ: empirical study / research in progress ไม่อ้างว่าเป็นวิธีใหม่หรือพร้อมตีพิมพ์

## เริ่มใช้งาน

ใช้ Python 3.11 ขึ้นไป แนะนำ virtual environment และ CPU สำหรับเริ่มต้น:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m unittest discover -v
python -m src.experiment --smoke --mode ssl --mask both --epochs 1 --pretrain-epochs 1 --output runs/smoke
```

ทดลองข้อมูลจริงหนึ่งงาน (สั่งดาวน์โหลด UCI HAR ประมาณ 60 MB และตรวจ checksum ก่อน):

```bash
python -m src.download_data
python -m src.experiment --mode supervised --fraction 0.1 --output runs/baseline
python -m src.experiment --mode ssl --mask both --fraction 0.1 --output runs/ssl
```

รันครบ 48 configurations: 4 วิธี × 4 สัดส่วน × 3 seeds ประเมิน 3 สภาพสัญญาณต่อ configuration รวม 144 evaluations:

```bash
python run_matrix.py --epochs 30 --pretrain-epochs 30
python -m src.summarize runs/full --output results/summary.csv
```

รองรับ `--device cuda` หรือ `--device mps` เมื่อเครื่องรองรับ ผลอาจต่างระหว่างอุปกรณ์ ต้องรายงาน environment ทุกครั้ง อย่าเปลี่ยนจำนวน epochs หรืออุปกรณ์แล้ว resume ในโฟลเดอร์เดิม: ใช้ `--runs runs/new_protocol` แทน

## ข้อมูลและวิธี

ใช้ UCI HAR inertial signals 6 ช่อง: total acceleration XYZ และ gyroscope XYZ ไม่ใช้ 561 handcrafted features และไม่ใช้ derived body acceleration เพื่อหลีกเลี่ยงช่องซ้ำเมื่อจำลอง sensor outage แต่ละหน้าต่างมี 128 samples (2.56 วินาทีที่ 50 Hz) ซึ่งชุดข้อมูลจัดเตรียมไว้แล้ว

ใช้ official test subjects; กัน validation subjects 4 คนจาก official train ด้วย split seed 2026 ผู้ใช้ไม่ซ้ำกันระหว่างชุด Standardization คำนวณจาก train เท่านั้น ป้ายกำกับ validation ใช้ครบทุกตัวอย่างเพื่อเลือก checkpoint ดังนั้นเปอร์เซ็นต์ป้ายหมายถึง **training labels เท่านั้น** ไม่ใช่งบ annotation รวม การสุ่มแบบ stratified ต้องใช้ป้ายเดิมเพื่อจำลอง label scarcity; ไม่ได้จำลองกระบวนการเลือกข้อมูลไร้ป้ายจริง

Encoder: Conv1d 6→32→64 + GroupNorm + GELU; mean pooling + linear head; SSL decoder เป็น Conv1d 64→6 ฝึก MSE เฉพาะตำแหน่งที่ปิดบัง

- supervised: เริ่มจาก random encoder
- time: ปิดช่วงเวลาต่อเนื่อง 25% ของหน้าต่าง
- channel: ปิดหนึ่งช่องจากหกช่อง
- both: รวมสองแบบ (อัตราปิดบังรวมสูงกว่า จึงยังแยกผลชนิด mask กับปริมาณ mask ไม่ได้)

SSL ใช้ train signals ทุกตัวอย่างโดยไม่อ่านป้ายเป็น target; fine-tune ทุก layer; Adam lr=0.001, batch=128, fixed epochs; เลือก checkpoint ด้วย validation Macro-F1 บนข้อมูลปกติเท่านั้น เมล็ดสุ่ม 0,1,2 ใช้ labeled subset เดียวกันข้ามวิธีและ nested subsets ข้ามสัดส่วน บันทึกจำนวนจริงเพราะปัดขึ้นต่อคลาส

Test conditions: none, acc (ปิดทั้ง 3 แกนความเร่ง), gyro (ปิดทั้ง 3 แกนไจโร) ค่าแทนที่เป็น 0 หลัง normalization = ค่าเฉลี่ย train ไม่ใช่ศูนย์เชิงกายภาพ เป็น stress test แบบจำลอง ไม่ใช่การทดสอบ hardware failure จริง และ dropout ทั้งกลุ่มแรงกว่าการปิดหนึ่งช่องขณะ pretraining

## ผลลัพธ์และการทำซ้ำ

ทุก run มี metadata.json, history.json, results.json, best.pt; run ที่ pretrain ใหม่มี encoder.pt การ reuse encoder ตรวจ configuration และ data hash ผล smoke เป็นข้อมูลสังเคราะห์และถูกกันออกจาก summary โดยอัตโนมัติ ห้ามรายงานเป็นผลวิจัย

อ่าน `docs/PROTOCOL_TH.md`, `docs/PAPER_DRAFT.md`, และรายงานผลเมื่อมีใน `results/` โมเดลที่ส่งแยกจาก source ZIP สามารถใช้ร่วมกับ normalization ใน metadata ได้

## GitHub public

```bash
git init
git add .
git commit -m "Add reproducible sensor SSL experiments"
git branch -M main
# สร้าง empty Public repository ใน GitHub แล้วใช้ URL ของคุณ:
git remote add origin https://github.com/YOUR_ACCOUNT/sensor-ssl.git
git push -u origin main
```

ไม่มีการสร้าง GitHub repository ให้โดยอัตโนมัติ ไม่มีบัญชีผู้ใช้/สิทธิ์ GitHub ที่ยืนยันในแชตนี้ `.gitignore` กัน data, runs, weights และ secrets ไว้ เปิดผลสรุปที่ตรวจแล้วได้ อย่าเพิ่มข้อมูลส่วนบุคคล ผู้เขียนควรเลือก license ของโค้ดและตกลงกับอาจารย์ก่อนอนุญาตให้ผู้อื่นนำไปใช้ (public visibility ไม่เท่ากับ open-source license)

## Attribution

Dataset: Reyes-Ortiz et al. (2013), Human Activity Recognition Using Smartphones, UCI, DOI https://doi.org/10.24432/C54S4K ; CC BY 4.0 ตามหน้า UCI https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones . ดาวน์โหลดจากต้นฉบับ ไม่รวมข้อมูลดิบใน source archive

Related work (ยังไม่ได้ reproduce): TS-TCC https://arxiv.org/abs/2106.14112 ; TS2Vec https://arxiv.org/abs/2106.10466 . โค้ดนี้เป็น masked reconstruction baseline เขียนใหม่ ไม่ใช่ implementation ของสองบทความดังกล่าว

ข้อมูลที่ใช้รันในชุดส่งมอบดาวน์โหลดจาก teaching mirror ของ UCI HAR เพราะ origin ช้า ดู URL, SHA-256 และขอบเขตการตรวจใน `docs/DATA_PROVENANCE.json` หากต้องการทำซ้ำด้วย archive เดียวกันให้ดาวน์โหลด URL ดังกล่าวเป็น `data/uci_har.zip` แล้วคัดลอก provenance ไป `data/provenance.json`

## ชุดส่งมอบ

ดู `START_HERE_TH.md` เพื่ออ่านไฟล์ตามลำดับ ผลวิจัยจริงทั้งหมดอยู่ใน `results/` และวิธีสร้างรายงานซ้ำอยู่ใน `docs/REPRODUCE.md`
