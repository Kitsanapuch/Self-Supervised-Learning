# คำสั่งนำโปรเจกต์ขึ้น GitHub

คู่มือนี้ใช้ **sensor-ssl-source.zip** ชุดที่เตรียมสำหรับเผยแพร่ ประกอบด้วยโค้ด รายงาน กราฟ ผลทดลอง และ `repository.bundle` สำหรับเก็บประวัติ Git ของงานทดลอง โดยไม่มีข้อมูลดิบหรือน้ำหนักโมเดล

## 1. สร้าง repository ว่างบน GitHub

เปิด [github.com/new](https://github.com/new) ตั้งชื่อ `sensor-ssl` เลือก **Public** แล้วกด **Create repository** เว้นตัวเลือก README, `.gitignore` และ license ว่างไว้ เพราะไฟล์เหล่านี้อยู่ในโปรเจกต์แล้ว (license ของโค้ดยังต้องตกลงกับอาจารย์ก่อนเพิ่ม)

## 2. แตก ZIP และเปิด repository ในเครื่อง

ตัวอย่างนี้สมมติว่าไฟล์ ZIP อยู่ใน `~/Downloads` บน macOS หรือ Linux:

```bash
mkdir -p ~/Documents/sensor-ssl-upload
unzip -q ~/Downloads/sensor-ssl-source.zip -d ~/Documents/sensor-ssl-upload
cd ~/Documents/sensor-ssl-upload
git clone sensor-ssl/repository.bundle sensor-ssl-public
cd sensor-ssl-public
```

ถ้า ZIP อยู่ที่อื่น ให้แก้ตำแหน่งในบรรทัด `unzip` `git clone` จะกู้ประวัติที่ใช้ระบุ source commit ของการทดลอง จึงควรใช้วิธีนี้แทนการเริ่มประวัติใหม่ด้วย `git init`

## 3. ตรวจไฟล์ก่อนเผยแพร่

```bash
git status --short
git log --oneline -3
git ls-files 'data/**' 'runs/**' '*.pt' '*.pth' '.env*'
```

บรรทัดแรกควรว่าง (ไม่มีไฟล์เปลี่ยน) และบรรทัดสุดท้ายควรว่าง (ไม่มีข้อมูลดิบ น้ำหนักโมเดล หรือไฟล์สภาพแวดล้อมใน Git) ประวัติเดิมแสดงผู้สร้าง commit เป็น `Codex <codex@localhost>`; ประวัตินี้จะมองเห็นได้ใน public repository

## 4. เชื่อมกับ GitHub และ push

แทน `YOUR_USERNAME` ด้วยชื่อบัญชี GitHub ของคุณ ใช้ URL ของ repository ว่างที่เพิ่งสร้าง:

```bash
git remote set-url origin https://github.com/YOUR_USERNAME/sensor-ssl.git
git remote -v
git push -u origin main
```

จากนั้นเปิด `https://github.com/YOUR_USERNAME/sensor-ssl` เพื่อตรวจว่า README, กราฟ และไฟล์ใน `results/` แสดงครบ GitHub อาจขอให้ลงชื่อเข้าใช้หรือยืนยันตัวตนเมื่อ push

## 5. อัปเดตครั้งถัดไป

ตั้งชื่อและอีเมลผู้เขียนสำหรับ commit ใหม่ใน repository นี้ครั้งเดียว หากยังไม่ได้ตั้งค่า:

```bash
git config user.name "YOUR_NAME"
git config user.email "YOUR_EMAIL_OR_GITHUB_NOREPLY_EMAIL"
```

เมื่อแก้ไฟล์แล้ว ใช้คำสั่งต่อไปนี้จากโฟลเดอร์ `sensor-ssl-public`:

```bash
git status
git add -A
git diff --cached --stat
git commit -m "Update research project"
git push
```

ไม่ต้องอัปโหลด `sensor-ssl-complete.zip` หรือคัดลอก `data/` และ `runs/` เข้า repository ชุดเต็มเก็บโมเดลและบันทึกการรันไว้สำหรับใช้ส่วนตัว ขณะที่ผลสรุปและหลักฐานที่จำเป็นต่อการอ่านงานอยู่ใน `results/` แล้ว

อ้างอิง: [GitHub Docs — Adding locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
