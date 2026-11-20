# TFEX Pre-Market Quant Radar (Phase 1 & 2)

ระบบวิเคราะห์เชิงปริมาณ (Quantitative Analysis Engine) สำหรับตลาดอนุพันธ์ **TFEX SET50 Index Futures & Options** อัตโนมัติทุกเช้าก่อนตลาดเปิด ออกแบบมาเพื่อสร้าง **"แผนการเทรด SET50 ประจำวัน (ภาษาไทย)"** ที่นำไปใช้ตัดสินใจหน้างานได้ทันที พร้อมบันทึกข้อมูลสถิติ Realized Volatility สู่ Turso Database และส่งรายงานตรงเข้าสู่ **Telegram Channel/Chat** ควบคู่กับไฟล์กราฟอินเทอร์แอคทีฟ Plotly Dashboard (.html)

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
.
├── .github/
│   └── workflows/
│       └── premarket_radar.yml   # รันอัตโนมัติ 06:45 น. (จันทร์–ศุกร์) บน GitHub Actions
├── .env.example                  # ตัวอย่าง Environment Variables สำหรับรันทดสอบในเครื่อง
├── .gitignore                    # ป้องกันการ Commit secret หรือ cache
├── pre_market_radar.py           # สคริปต์หลักวิเคราะห์ Greeks, GEX, RV, Flow, Profile และส่งแผนเทรด
├── requirements.txt              # numpy, scipy, pandas, plotly, requests, python-dotenv
└── README.md                     # เอกสารแนะนำระบบและคู่มือการใช้งาน
```

---

## 🧭 เสาหลักของแผนการเทรด (5 Core Trading Pillars)

รายงานสรุปทุกเช้าจะแบ่งออกเป็น 5 หัวข้อสำคัญที่กระชับ ชัดเจน และปฏิบัติได้จริง (Actionable):

### 1. 🎯 กรอบวิ่งคาดการณ์วันนี้ (1-Day Expected Move)
* คำนวณระยะแกว่งตัวทางสถิติ 1 ส่วนเบี่ยงเบนมาตรฐาน (1-SD) อิงตามความผันผวนแฝง ATM (Black-76 Implied Volatility):
  $$\text{Expected Move}_{1D} = S50 \times \frac{\text{ATM\_IV}}{\sqrt{252}}$$
* แสดงผล:
  * **กรอบราคา 1-SD:** $\text{Lower 1-SD} - \text{Upper 1-SD}$ (ระยะแกว่งตัวคาดการณ์ $\pm \text{Points}$)
  * **ระดับความผันผวน (ATM IV):** ต่ำมาก / ปกติ / สูง

### 2. 🧱 แนวรับ - แนวต้านสำคัญรายวัน (Tactical Key Levels)
* วิเคราะห์ Open Interest (OI) และ Net GEX เฉพาะโซน Strike รอบราคาปัจจุบัน (**Spot $\pm$ 25 จุด**) เพื่อคัดเลือกแนวรับ-แนวต้านย่อยที่ใช้เทรดจริงระหว่างวัน:
  * 🔴 **ต้าน 2 / ต้าน 1:** Strike ที่มี Call OI หนาแน่น
  * 📍 **ราคาปิดล่าสุด / Max Pain:** ระดับราคา Settlement ล่าสุด และจุดดึงดูดของ Options
  * 🟢 **รับ 1 / รับ 2:** Strike ที่มี Put OI หนาแน่น
* ระบุกรอบใหญ่ไตรมาส (**Put Wall – Call Wall** และ **Gamma Flip**) ไว้อ้างอิง

### 3. 📊 โครงสร้าง Intraday & ความผันผวน (Warm-up: X/20 วัน)
* **Realized vs Implied Volatility:**
  * คำนวณ Log Return รายนาทีจากแท่งเทียน Intraday เพื่อหาค่า **Annualized Realized Volatility (RV)**:
    $$\text{RV}_{\text{annualized}} = \sqrt{252 \times \sum_{t=1}^M r_t^2}$$
  * บันทึกลงตาราง `daily_realized_vol` ใน Turso เพื่อสะสมชุดข้อมูลสำหรับโมเดล HAR-RV
  * เปรียบเทียบ **RV vs ATM IV** เพื่อระบุสภาวะตลาด (IV Premium สูง = เหมาะเทรดในกรอบ / RV ทะลุ IV = ระวัง Breakout)
* **Intraday Volume Profile (Rolling 20 วันทำการ):**
  * คำนวณหา **POC (Point of Control)**, **VAH (Value Area High)** และ **VAL (Value Area Low)** ครอบคลุม 70% ของปริมาณการซื้อขายสะสม
  * จับคู่ Confluence Zones กับแนวรับ-ต้าน Options (Call/Put Wall, Max Pain) ในระยะ $\pm 3$ จุด เพื่อระบุแนวปะทะคอนกรีต
* **พฤติกรรมราคา & Basis ท้ายตลาด (16:30 – 16:55 น.):**
  * ตรวจจับการเปลี่ยนแปลงของราคา Futures และค่า Basis ($\text{Futures} - \text{Spot}$) ในช่วง 25 นาทีสุดท้าย เพื่อประเมินแรงไล่ราคาหรือแรงทิ้งสัญญาก่อนตลาดปิด

### 4. 🌊 พฤติกรรมรายใหญ่ (Smart Money Flow & Footprint)
* **การเคลื่อนไหวรายวัน (Intraday Smart Money Flow):**
  * แปลงผลนักลงทุนต่างชาติ (**Foreign Net Flow**) เทียบกับอัตราการเปลี่ยนแปลงของสถานะคงค้างรวม (**$\Delta$OI**):
    * 🟢 **New Long Build-up:** ต่างชาติซื้อสุทธิ + OI เพิ่มขึ้น (สะสมพลังขาขึ้นแท้จริง)
    * 🟡 **Short Covering:** ต่างชาติซื้อสุทธิ + OI ลดลง (ปิดสัญญา Short เก่า ระวัง Technical Rebound สั้น)
    * 🔴 **New Short Build-up:** ต่างชาติขายสุทธิ + OI เพิ่มขึ้น (เปิด Short ใหม่ กดดันขาลง)
    * 🟠 **Long Liquidation:** ต่างชาติขายสุทธิ + OI ลดลง (ปิดทำกำไร/ลดพอร์ต Long)
  * สรุปยอดซื้อขายสุทธิครบทุกกลุ่ม: ต่างชาติ, สถาบัน, รายย่อย
* **แกะรอยพอร์ตสะสมและต้นทุนฝรั่ง (Foreign Cumulative Footprint 60 วัน):**
  * **ยอดสะสมสุทธิ & ต้นทุนเฉลี่ยสะสม (Foreign Cost Basis):** คำนวณแบบ Running Inventory ถ่วงน้ำหนักยอดซื้อขายจริงกับราคา Settlement เพื่อหาต้นทุนเฉลี่ยที่แท้จริง พร้อมประเมินสถานะ กำไร/ขาดทุน
  * **20-Day Flow Z-Score:** หาก $|Z| \ge 1.5$ จะแจ้งเตือนยอดซื้อ/ขายผิดปกติอย่างมีนัยสำคัญ
  * **Price-Flow Divergence (5 วัน):** ตรวจจับสัญญาณขัดแย้ง เช่น ราคาขึ้นแต่ฝรั่งเทขาย หรือราคาย่อแต่ฝรั่งดักช้อน
* **Options Strike Footprint:**
  * ตรวจจับยอด Options สุทธิของกลุ่มต่างชาติ (Call/Put)
  * คัดกรอง Strike ที่มี $\Delta$OI สูงสุด พร้อมทิศทาง $\Delta$IV เพื่อระบุเจตนาสถาบัน (Aggressive Buying, Writing / Strike Cap, Downside Hedging, Floor Support)

### 5. ⚔️ กลยุทธ์และแผนการเล่นวันนี้ (Daily Action Plan)
* **สภาวะตลาด (Dealer Gamma & Institutional Cost Line):**
  * **Gamma Flip & Net GEX Regime:** ระบุสภาวะตลาดระหว่าง Positive Gamma (ลดความผันผวน / Mean Reversion) และ Negative Gamma (เร่งความผันผวน / Trend Expansion)
  * **Institutional Defense Line:** วิเคราะห์ความสัมพันธ์ของราคาปัจจุบันกับต้นทุนเฉลี่ยสะสมฝรั่ง เพื่อใช้เป็นแนวรับ/แนวต้านจิตวิทยา
* **แผนปฏิบัติการหน้างาน:**
  * กลยุทธ์หลักประจำวัน
  * แผนฝั่ง Long (จุดตั้งรับ, เงื่อนไขยืนยัน, เป้าทำกำไร)
  * แผนฝั่ง Short (จุดดัก Short, เงื่อนไขยืนยัน, เป้าทำกำไร)
  * จุดตัดขาดทุน (Stop Loss) อัตโนมัติ ผสานแนวรับ-ต้าน Tactical และ Institutional Defense Level

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
   pip install -r requirements.txt      # pin ไว้แล้ว (resolve บน Python 3.11)
   ```

2. **รันชุดทดสอบ (unit tests):**
   ```bash
   pip install -r requirements-dev.txt  # requirements.txt + pytest
   python -m pytest -q                  # คาดหวัง 22 passed
   ```

3. **ทดสอบรันด้วยชุดข้อมูลจำลอง (Mock Test Mode):**
   ```bash
   python pre_market_radar.py --test
   ```
   *(โหมดนี้จะคำนวณและแสดงผลตัวอย่างรายงานแผนเทรดภาษาไทยบนหน้าจอ และหากมีการตั้งค่า TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID จะส่งเข้า Telegram ให้ทันที — ตั้งแต่รอบ review 2026-11-22 โหมด `--test` ทำหน้าที่เป็น **smoke test**: ถ้าผลลัพธ์ขาดส่วนสำคัญหรือไม่ได้เขียน dashboard จะ `exit 1`)*

4. **รันจริงเชื่อมต่อฐานข้อมูล Turso:**
   ```bash
   python pre_market_radar.py --force
   ```

---

## 🧩 โค้ดที่ใช้ร่วมกับ repo อื่น (tfex-common)

ปฏิทิน SET / expiry, THOR unit handling, Turso client และ Telegram dispatch ไม่ได้อยู่ในไฟล์นี้แล้ว — import จาก [`tfex-common`](https://github.com/met37818-stack/tfex-common) (ปักหมุด tag ใน `requirements.txt`) เพื่อให้ทั้ง 4 repo ใช้โค้ดชุดเดียวกัน

## ⚠️ ข้อกำหนดด้านข้อมูล (Fail-Loud Rules)

- **THOR ต้องเป็น percent p.a. เท่านั้น** — ระบบหาร 100 แบบไม่มีเงื่อนไข (เดิมเดาจากขนาด ทำให้ THOR 0.45% อ่านเป็น 45% แล้ว IV ที่แก้ได้เพี้ยนไปหลาย vol point) และจะ `raise` ถ้าค่าเกิน `THOR_MAX_PCT = 25.0`
- **ห้ามสรุปทิศทางจากข้อมูลที่ขาด** — ถ้าคำนวณ 25-Delta IV ได้เพียงฝั่งเดียว `skew_25d` จะเป็น `None` และขึ้น `N/A` (ไม่ใช่ 0.0) และถ้า delta-IV เปลี่ยนไม่ได้ จะรายงาน "งดสรุปเจตนา" แทนการเดาว่าเป็น Call/Put Writing
- **Turso ล่มชั่วคราวไม่ทำให้ job ตาย** — retry network error และ HTTP 429/5xx (backoff, honour `Retry-After` cap 30s); SQL error ไม่ retry
- **Telegram ล่มไม่ทำให้ token หลุด** — exception ถูก log เฉพาะชื่อคลาส ไม่ log URL ที่มี bot token
