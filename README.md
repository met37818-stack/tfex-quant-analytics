# TFEX Pre-Market Quant Radar (Phase 1)

ระบบวิเคราะห์เชิงปริมาณ (Quantitative Analysis Engine) สำหรับตลาดอนุพันธ์ **TFEX SET50 Index Futures & Options** อัตโนมัติทุกเช้าก่อนตลาดเปิด ออกแบบมาเพื่อสร้าง **"แผนการเทรด SET50 ประจำวัน (ภาษาไทย)"** ที่นำไปใช้ตัดสินใจหน้างานได้ทันที และส่งรายงานตรงเข้าสู่ **Telegram Channel/Chat** พร้อมไฟล์กราฟอินเทอร์แอคทีฟ Plotly Dashboard (.html)

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
.
├── .github/
│   └── workflows/
│       └── premarket_radar.yml   # รันอัตโนมัติ 06:45 น. (จันทร์–ศุกร์) บน GitHub Actions
├── .env.example                  # ตัวอย่าง Environment Variables สำหรับรันทดสอบในเครื่อง
├── .gitignore                    # ป้องกันการ Commit secret หรือ cache
├── pre_market_radar.py           # สคริปต์หลักวิเคราะห์ Quant Greeks, GEX, Flow และส่งแผนเทรด
├── requirements.txt              # numpy, scipy, pandas, plotly, requests, python-dotenv
└── README.md                     # เอกสารแนะนำระบบและคู่มือการใช้งาน
```

---

## 🧭 เสาหลักของแผนการเทรด (4 Core Trading Pillars)

รายงานสรุปทุกเช้าจะแบ่งออกเป็น 4 หัวข้อสำคัญที่กระชับและปฏิบัติได้จริง (Actionable):

### 1. 🎯 กรอบวิ่งคาดการณ์วันนี้ (1-Day Expected Move)
* คำนวณระยะแกว่งตัวทางสถิติ 1 ส่วนเบี่ยงเบนมาตรฐาน (1-SD) อิงตามความผันผวนแฝง ATM (Black-76 Implied Volatility):
  $$\text{Expected Move}_{1D} = S50 \times \frac{\text{ATM\_IV}}{\sqrt{252}}$$
* แสดงผล:
  * **กรอบบน (Upper 1-SD):** $S50 + \text{Expected Move}_{1D}$
  * **กรอบล่าง (Lower 1-SD):** $S50 - \text{Expected Move}_{1D}$
  * **ระยะแกว่งตัวคาดการณ์ (1-SD):** $\pm \text{Points}$
  * **ระดับความผันผวน (ATM IV):** ต่ำมาก / ปกติ / สูง

### 2. 🧱 แนวรับ - แนวต้านสำคัญรายวัน (Tactical Key Levels)
* วิเคราะห์ Open Interest (OI) และ Net GEX เฉพาะโซน Strike รอบราคาปัจจุบัน (**Spot $\pm$ 25 จุด**) เพื่อคัดเลือกแนวรับ-แนวต้านย่อยที่ใช้เทรดจริงระหว่างวัน แทนที่จะดูเพียงกรอบใหญ่ทั้งไตรมาส:
  * 🔴 **ต้าน 2:** Strike ที่มี Call OI หนาแน่นระดับบน
  * 🔴 **ต้าน 1:** Strike แนวต้านแรกใกล้ราคา
  * 🎯 **Max Pain:** ระดับราคาที่มีมูลค่าการจ่ายเงินสุทธิของ Options ต่ำสุด (จุดดึงดูด)
  * 📍 **ราคาปิดล่าสุด:** ราคา Settlement ของสัญญา Front Quarter ล่าสุด
  * 🟢 **รับ 1:** Strike แนวรับแรกใกล้ราคา
  * 🟢 **รับ 2:** Strike ที่มี Put OI หนาแน่นรองรับ
* ระบุกรอบใหญ่ไตรมาส (**Put Wall – Call Wall** และ **Gamma Flip**) ไว้ด้านล่าง

### 3. 🌊 พฤติกรรมรายใหญ่ (Smart Money Flow)
* แปลงผลการเคลื่อนไหวของนักลงทุนต่างชาติ (**Foreign Net Flow**) เทียบกับอัตราการเปลี่ยนแปลงของสถานะคงค้างรวม (**$\Delta$OI**):
  * 🟢 **New Long Build-up:** ต่างชาติซื้อสุทธิ + OI เพิ่มขึ้น (สะสมพลังขาขึ้นแท้จริง)
  * 🟡 **Short Covering:** ต่างชาติซื้อสุทธิ + OI ลดลง (ปิดสัญญา Short เก่า ระวัง Technical Rebound สั้น)
  * 🔴 **New Short Build-up:** ต่างชาติขายสุทธิ + OI เพิ่มขึ้น (เปิด Short ใหม่ กดดันขาลง)
  * 🟠 **Long Liquidation:** ต่างชาติขายสุทธิ + OI ลดลง (ปิดทำกำไร/ลดพอร์ต Long)
* สรุปยอดซื้อขายสุทธิครบทุกกลุ่ม: Foreign, Institutional, Local Retail

### 4. ⚔️ กลยุทธ์และแผนการเล่นวันนี้ (Daily Action Plan)
* **Gamma Flip & Net GEX Consistency:**
  * หาก $S50 \ge \text{Gamma Flip}$: อยู่ในโซน **Positive Gamma** (ดีลเลอร์ช่วยซับแรงกระแทก ตลาดมักแกว่งในกรอบ Mean Reversion)
  * หาก $S50 < \text{Gamma Flip}$: อยู่ในโซน **Negative Gamma** (ดีลเลอร์เทรดตามโมเมนตัมเร่งแรงแกว่งตัว ระวังหลุดแนวรับแล้วรูดแรง Trend Expansion)
* **แผนปฏิบัติการหน้างาน:**
  * กลยุทธ์หลัก
  * แผนฝั่ง Long (จุดเข้า, เงื่อนไข, เป้าหมายกำไร)
  * แผนฝั่ง Short (จุดเข้า, เงื่อนไข, เป้าหมายกำไร)
  * จุดตัดขาดทุน (Stop Loss) อัตโนมัติ

---

## 📊 รายงานอินเทอร์แอคทีฟ (Plotly Dashboard)

ระบบจะแนบไฟล์รายงาน `.html` (3 Subplots) ส่งคู่กันใน Telegram:
1. **Net Gamma Exposure (GEX) Profile by Strike:** แสดงแท่ง GEX ราย Strike พร้อมเส้นปะระดับราคาปัจจุบัน, Gamma Flip, Call Wall, Put Wall และ Upper/Lower 1-SD
2. **Implied Volatility Smile & Skew Curve:** กราฟความผันผวน Call/Put ของ Front Quarter เทียบกับ Next Quarter
3. **Open Interest Distribution & Max Pain:** แท่งเปรียบเทียบ Call OI vs Put OI และเส้น Max Pain

---

## 🔐 การตั้งค่า Secrets บน GitHub Actions

| Secret Name | คำอธิบาย |
| :--- | :--- |
| `TURSO_DATABASE_URL` | URL ของ Turso Database (`libsql://...`) |
| `TURSO_AUTH_TOKEN` | Auth Token สำหรับเข้าถึง Turso |
| `TELEGRAM_BOT_TOKEN` | Token บอท Telegram สำหรับแจ้งเตือน |
| `TELEGRAM_CHAT_ID` | Chat ID ปลายทางที่รับรายงาน |

---

## 🚀 วิธีการทดสอบในเครื่อง (Local Run)

1. **ติดตั้ง Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **ทดสอบรันด้วยชุดข้อมูลจำลอง (Mock Test Mode):**
   ```bash
   python pre_market_radar.py --test
   ```
   *(โหมดนี้จะคำนวณและแสดงผลตัวอย่างรายงานแผนเทรดภาษาไทยบนหน้าจอ และหากมีการตั้งค่า TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID จะส่งเข้า Telegram ให้ทันที)*

3. **รันจริงเชื่อมต่อฐานข้อมูล Turso:**
   ```bash
   python pre_market_radar.py --force
   ```
