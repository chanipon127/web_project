import json
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2,psycopg2.extras
from datetime import datetime
import bcrypt
import os
import re
import shutil
import pandas as pd
import numpy as np
from fastapi import Query
from typing import List
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
from ai_scoring import evaluate_single_answer


app = FastAPI()

# 🔓 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🌐 Database Connection
conn = psycopg2.connect(
    host="ep-cold-bonus-adrb2tv4-pooler.c-2.us-east-1.aws.neon.tech",
    database="neondb",
    user="neondb_owner",
    password="npg_Hi8SPj1WXrds",
    port=5432
)

# 📌 Schema
class RegisterForm(BaseModel):
    username: str
    fullname: str
    password: str
    role: str

# 🔐 Register
@app.post("/api/register")
async def register_user(data: RegisterForm):
    try:
        hashed_password = bcrypt.hashpw(data.password.encode('utf-8'), bcrypt.gensalt())
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO users (role, username, fullname, password, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (data.role, data.username, data.fullname, hashed_password.decode('utf-8'), datetime.now()))
        conn.commit()
        return {"message": "สมัครสมาชิกสำเร็จ"}
    except Exception as e:
        conn.rollback()
        return {"message": f"เกิดข้อผิดพลาด: {str(e)}"}

# 📌 Schema
class LoginForm(BaseModel):
    username: str
    password: str
    
# ✅ Login
@app.post("/api/login")
async def login(data: LoginForm):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT password, role, fullname FROM users WHERE username = %s", (data.username,))
            result = cur.fetchone()
            if result:
                db_password, role, fullname = result
                if bcrypt.checkpw(data.password.encode('utf-8'), db_password.encode('utf-8')):
                    return {
                        "message": "เข้าสู่ระบบสำเร็จ",
                        "username": data.username,
                        "fullname": fullname,
                        "role": role
                    }
        raise HTTPException(status_code=401, detail="ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")
    
# 📌 Schema
class ContactForm(BaseModel):
    name: str
    user: str
    message: str

# ✉️ Contact Admin API
@app.post("/api/contact-admin")
async def admin_contact(data: ContactForm):
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO admin_contact (name, username, message, created_at)
            VALUES (%s, %s, %s, %s)
        """, (data.name, data.user, data.message, datetime.now()))
        conn.commit()
        return {"message": "ส่งข้อความถึงผู้ดูแลระบบเรียบร้อยแล้ว"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

# 📂 ensure uploads folder
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 📌 Schema
class UserInfo(BaseModel):
    fullname: str
    username: str
    role: str
    profile_img: str = "https://via.placeholder.com/120"


# ✅ API: ดึงข้อมูลผู้ login
@app.get("/api/userinfo", response_model=UserInfo)
async def get_userinfo(username: str):
    cur = conn.cursor()
    cur.execute("SELECT fullname, role, profile_img FROM users WHERE username = %s", (username,))
    result = cur.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้")
    
    fullname, role, profile_img = result
    return {
        "fullname": fullname,
        "username": username,
        "role": role,
        "profile_img": profile_img or "https://via.placeholder.com/120"
    }

# ✅ Update User
@app.post("/api/update_user")
async def update_user(
    username: str = Form(...),  # เอามาจาก localStorage
    new_username: str = Form(None),   # username ใหม่
    fullname: str = Form(None),
    password: str = Form(None),
    profile_img: UploadFile = File(None)
):
    try:
        with conn.cursor() as cur:
            updates = []
            values = []

            if new_username:
                updates.append("username = %s")
                values.append(new_username)
                
            if fullname:
                updates.append("fullname = %s")
                values.append(fullname)

            if password:
                hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                updates.append("password = %s")
                values.append(hashed_password)

            if profile_img:
                upload_dir = "uploads"
                os.makedirs(upload_dir, exist_ok=True)

                # ป้องกันชื่อไฟล์ชนกัน
                import uuid
                filename = f"{uuid.uuid4().hex}_{profile_img.filename}"
                file_path = os.path.join(upload_dir, filename)

                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(profile_img.file, buffer)

                file_url = f"http://127.0.0.1:8000/uploads/{filename}"
                updates.append("profile_img = %s")
                values.append(file_url)

            if not updates:
                raise HTTPException(status_code=400, detail="ไม่มีข้อมูลใหม่สำหรับอัปเดต")

            values.append(username)
            sql = f"UPDATE users SET {', '.join(updates)} WHERE username = %s"
            cur.execute(sql, tuple(values))

            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้สำหรับอัปเดต")

            conn.commit()

        # ถ้าเปลี่ยน username ต้องอัปเดต localStorage ด้วย
        return {
            "message": "อัปเดตข้อมูลสำเร็จ",
            "new_username": new_username if new_username else username
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

# 📌 API: ดึงผู้ใช้งานทั้งหมด
@app.get("/api/users")
async def get_users():
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT username, fullname, role FROM users ORDER BY created_at DESC")
            rows = cur.fetchall()

        users = []
        for row in rows:
            username, fullname, role = row
            users.append({
                "username": username,
                "fullname": fullname,
                "role": role
            })

        return {"users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

#ลบผู้ใช้งาน
@app.delete("/api/delete_user")
async def delete_user(username: str):
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username = %s", (username,))
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้งาน")
            conn.commit()
        return {"message": f"ลบผู้ใช้งาน {username} สำเร็จ"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

# 🔹 ดึงปีการศึกษาไม่ซ้ำ
@app.get("/exam_years", response_model=List[int])
def get_exam_years():
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT exam_year FROM exam ORDER BY exam_year DESC')
    years = [row[0] for row in cursor.fetchall()]
    return years


# 🔹 ดึงระดับชั้นเรียนไม่ซ้ำ
@app.get("/group_ids", response_model=List[str])
def get_group_ids():
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT group_id FROM exam
        UNION
        SELECT DISTINCT group_id FROM answer
        ORDER BY group_id
    """)
    groups = [row[0] for row in cursor.fetchall()]
    return groups

# 📌 ดึง feedback ทั้งหมด
@app.get("/api/contact-admin-all")
async def get_all_contacts(): 
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT contact_id, username, name, message, created_at
            FROM admin_contact
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        # ✅ แปลง tuple → dict
        feedback_list = [
            {
                "contact_id": r[0],
                "username": r[1],
                "name": r[2],
                "message": r[3],
                "created_at": r[4]
            }
            for r in rows
        ]

        return feedback_list
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ ลบ feedback ------------------
@app.delete("/api/contact-admin/{contact_id}")
async def delete_feedback(contact_id: int):
    try:
        #conn = get_connection()
        cur = conn.cursor()
        # ตรวจสอบว่ามี contact_id นี้ไหม
        cur.execute("SELECT contact_id FROM admin_contact WHERE contact_id = %s", (contact_id,))
        if cur.fetchone() is None:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="ไม่พบข้อความนี้")

        # ลบข้อความ
        cur.execute("DELETE FROM admin_contact WHERE contact_id = %s", (contact_id,))
        conn.commit()
        cur.close()
        conn.close()

        return {"message": "ลบข้อความเรียบร้อย"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- Pydantic Model -----------------
class Answer(BaseModel):
    student_id: int
    group_id: str
    exam_year: int
    essay_text: str
    essay_analysis: str
    status: str



# -----------------------
# POST เพิ่มคำตอบ(เป็นไฟล์)
@app.post("/api/answers/upload")
async def upload_answers(file: UploadFile = File(...)):
    cursor = None
    try:
        # ตรวจไฟล์
        if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx")):
            raise HTTPException(status_code=400, detail="รองรับเฉพาะ CSV หรือ Excel")

        # อ่าน DataFrame
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)

        # ตรวจว่ามีคอลัมน์พื้นฐาน
        base_cols = {"student_id", "group_id", "exam_year", "essay_text", "essay_analysis"}
        if not base_cols.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail=f"ต้องมีคอลัมน์พื้นฐาน: {base_cols}"
            )

        cursor = conn.cursor()
        inserted = 0

        # เตรียมชื่อคอลัมน์คะแนน
        score_cols_t1 = [f"score_s{i}_t1" for i in range(1, 14)]
        score_cols_t2 = [f"score_s{i}_t2" for i in range(1, 14)]
        score_cols = score_cols_t1 + score_cols_t2

        for _, row in df.iterrows():
            # --- ตรวจค่า student_id และ exam_year ---
            student_id_val = row.get("student_id")
            if pd.isna(student_id_val) or str(student_id_val).strip() == "":
                continue  # ข้ามแถวที่ student_id ว่าง

            # ✅ แปลงให้เป็น string ไม่มี .0
            if isinstance(student_id_val, (int, np.integer)):
                student_id_val = str(student_id_val)
            elif isinstance(student_id_val, float):
                student_id_val = str(int(student_id_val))
            else:
                student_id_val = str(student_id_val).strip()

            # ✅ แปลง exam_year ให้เป็น int
            if pd.isna(row["exam_year"]):
                continue
            exam_year_val = int(row["exam_year"])

            group_id_val = row["group_id"]

            # --- ตรวจว่าแถวนี้มีอยู่แล้วใน DB ---
            cursor.execute("""
                SELECT 1 FROM answer
                WHERE student_id=%s AND exam_year=%s AND group_id=%s
            """, (student_id_val, exam_year_val, group_id_val))
            if cursor.fetchone():
                continue  # มีอยู่แล้ว → ข้าม

            # --- insert ตาราง answer ---
            cursor.execute("""
                INSERT INTO answer (student_id, group_id, exam_year, essay_text, essay_analysis, status)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                student_id_val,
                group_id_val,
                exam_year_val,
                str(row["essay_text"]),
                str(row["essay_analysis"]),
                "ยังไม่ได้ตรวจ"
            ))

            # --- เตรียมคะแนน ---
            scores = []
            for col in score_cols:
                val = row.get(col, None)
                if pd.isna(val):
                    scores.append(None)
                else:
                    try:
                        scores.append(float(val))
                    except ValueError:
                        scores.append(None)

            # --- insert ตาราง teacher_score ---
            cursor.execute(f"""
                INSERT INTO teacher_score (
                    student_id, exam_year, group_id, {','.join(score_cols)}
                )
                VALUES (%s, %s, %s, {','.join(['%s']*len(score_cols))})
            """, [student_id_val, exam_year_val, group_id_val] + scores)

            inserted += 1

        conn.commit()
        return {"message": "อัปโหลดสำเร็จ", "inserted": inserted}

    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()



# -----------------------
# POST เพิ่มคำตอบ (แบบเดี่ยว)
# -----------------------
@app.post("/api/answers")
async def create_answer(answer: Answer):
    cursor = None
    try:
        cursor = conn.cursor()

        # แปลง student_id เป็น string (เพราะใน DB เป็น VARCHAR)
        student_id_val = str(answer.student_id).strip()

        # ตรวจว่ามี record ซ้ำหรือยัง (student_id + exam_year + group_id)
        cursor.execute("""
            SELECT 1 FROM answer
            WHERE student_id=%s AND exam_year=%s AND group_id=%s
        """, (student_id_val, answer.exam_year, answer.group_id))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="คำตอบนี้มีอยู่แล้ว")

        # insert ตาราง answer
        cursor.execute("""
            INSERT INTO answer (student_id, group_id, exam_year, essay_text, essay_analysis, status)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            student_id_val,
            answer.group_id,
            answer.exam_year,
            answer.essay_text,
            answer.essay_analysis,
            answer.status or "ยังไม่ได้ตรวจ"
        ))

        # insert ตาราง teacher_score (สร้างค่าว่างไว้ก่อน)
        score_cols_t1 = [f"score_s{i}_t1" for i in range(1, 14)]
        score_cols_t2 = [f"score_s{i}_t2" for i in range(1, 14)]
        score_cols = score_cols_t1 + score_cols_t2

        cursor.execute(f"""
            INSERT INTO teacher_score (student_id, exam_year, group_id, {','.join(score_cols)})
            VALUES (%s, %s, %s, {','.join(['%s']*len(score_cols))})
        """, [student_id_val, answer.exam_year, answer.group_id] + [None]*len(score_cols))

        conn.commit()
        return {"message": "เพิ่มคำตอบสำเร็จ", "student_id": student_id_val}

    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()



# GET ดึงคำตอบทั้งหมด
@app.get("/api/answers-all")
def get_all_answers():
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT answer_id, student_id, exam_year, essay_text, essay_analysis, group_id, status
            FROM answer
            ORDER BY answer_id DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        results = []
        for r in rows:
            results.append({
                "answer_id": r[0],
                "student_id": r[1],
                "exam_year": r[2],
                "essay_text": r[3],
                "essay_analysis": r[4],
                "group_id": r[5],
                "status": r[6]
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()


# 🔹 Pydantic Model
# -------------------------------
class Answer(BaseModel):
    student_id: int
    group_id: str
    exam_year: int
    essay_text: str
    essay_analysis: str
    status: str

# -------------------------------
# ✅ API: ดึงคำตอบทั้งหมด
# -------------------------------
@app.get("/api/answers-all")
def get_all_answers():
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT answer_id, student_id, exam_year, essay_text, essay_analysis, group_id, status, score
            FROM answer
            ORDER BY answer_id DESC
        """)
        rows = cursor.fetchall()

        results = []
        for r in rows:
            results.append({
                "answer_id": r[0],
                "student_id": r[1],
                "exam_year": r[2],
                "essay_text": r[3],
                "essay_analysis": r[4],
                "group_id": r[5],
                "status": r[6],
                "score": r[7] if r[7] is not None else None
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()


# ✅ API: ตรวจคำตอบด้วย AI
# ✅ API: ตรวจคำตอบด้วย AI (เวอร์ชันสุดท้ายที่แก้ไขตาม Log จริง)
@app.post("/api/check-answer/{answer_id}")
async def check_answer(answer_id: int):
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT essay_text, essay_analysis FROM answer WHERE answer_id = %s", (answer_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="ไม่พบคำตอบ")

        essay_text, essay_analysis = row

        # 1. เรียก AI และรับผลลัพธ์เป็น Dictionary
        ai_result_dict = evaluate_single_answer(essay_text, essay_analysis)
        
        # Log ผลดิบไว้เผื่อตรวจสอบในอนาคต
        print("AI raw result for answer_id", answer_id, ":", json.dumps(ai_result_dict, indent=2, ensure_ascii=False))

        # 2. ฟังก์ชันแปลงผลลัพธ์สำหรับโครงสร้าง JSON แบบ "แบน" (Flat)
        def map_ai_results_to_s_format(results):
            formatted_desc = {}

            # *** สร้าง Mapping ของ Key ที่ถูกต้อง 100% จาก Log ***
            # หมายเหตุ: s3 มี key ที่ผิดปกติจาก AI แต่เราต้องใช้ตามนั้น
            key_mapping = {
                "s1": "ข้อที่ 1 - ใจความสำคัญ",
                "s2": "ข้อที่ 1 - การเรียงลำดับและเชื่อมโยงความคิด",
                "s3": "ข้อที่ 1 - ความถูกต้องตามหลักการเขียนย่อความ",
                "s4": "ข้อที่ 1 - การสะกดคำ",
                "s5": "ข้อที่ 1 - การใช้คำ/ถ้อยคำสำนวน",
                "s6": "ข้อที่ 1 - การใช้ประโยค",
                "s7": "ข้อที่ 2 - คำบอกข้อคิดเห็น",
                "s8": "ข้อที่ 2 - เหตุผลสนับสนุน",
                "s9": "ข้อที่ 2 - การเรียงลำดับและเชื่อมโยงความคิด",
                "s10": "ข้อที่ 2 - ความถูกต้องตามหลักการแสดงความคิดเห็น",
                "s11": "ข้อที่ 2 - การสะกดคำ/การใช้ภาษา",
                "s12": "ข้อที่ 2 - การใช้คำ/ถ้อยคำสำนวน",
                "s13": "ข้อที่ 2 - การใช้ประโยค",
            }
            
            # วน Loop เพื่อดึงค่าของแต่ละ S
            for s_key, ai_key in key_mapping.items():
                # ดึงข้อมูลจาก top-level dictionary โดยตรง
                data = results.get(ai_key, {}) 
                
                # AI อาจจะส่ง score มาใน key ชื่อ 'score' หรือ 'คะแนน'
                score = data.get("score", data.get("คะแนน", 0.0))
                
                # ใช้ 'details' เป็น feedback ถ้ามี, ถ้าไม่มีก็ใช้ object ทั้งหมด
                feedback_data = data.get("details", data)
                
                formatted_desc[s_key] = {
                    "score": float(score),
                    "feedback": json.dumps(feedback_data, ensure_ascii=False)
                }
            
            # ดึงคะแนนรวมทั้งหมด
            total_score_key = "คะแนนรวมทั้งหมด (30 คะแนน)"
            total_score = results.get(total_score_key, 0.0)

            return formatted_desc, float(total_score)

        # 3. เรียกใช้ฟังก์ชันแปลงค่า
        formatted_description, total_score = map_ai_results_to_s_format(ai_result_dict)
        
        # 4. บันทึกลงฐานข้อมูล
        cursor.execute("""
            UPDATE answer
            SET score=%s,
                status='ตรวจแล้ว',
                description=%s
            WHERE answer_id = %s
        """, (total_score, json.dumps(formatted_description, ensure_ascii=False), answer_id))
        
        conn.commit()

        return {
            "message": "ตรวจคำตอบสำเร็จ",
            "score": total_score,
            "description": formatted_description
        }

    except Exception as e:
        if conn:
            conn.rollback()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")
    finally:
        if cursor:
            cursor.close()


# -----------------------------
# API: ดูผลคำตอบ + คะแนนครู (เวอร์ชันปรับปรุง)
# -----------------------------
@app.get("/api/view-score/{answer_id}")
def view_score(answer_id: int):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # ดึงคำตอบหลัก
        cur.execute("""
            SELECT answer_id, student_id, group_id, exam_year,
                   essay_text, essay_analysis, status, score, description
            FROM answer
            WHERE answer_id = %s
        """, (answer_id,))
        answer = cur.fetchone()
        if not answer:
            raise HTTPException(status_code=404, detail="ไม่พบคำตอบนี้")

        # ดึงคะแนนครู (teacher_score)
        cur.execute("""
            SELECT * FROM teacher_score
            WHERE student_id = %s AND exam_year = %s AND group_id = %s
        """, (answer["student_id"], answer["exam_year"], answer["group_id"]))
        teacher_row = cur.fetchone()

        teacher_scores = {"teacher1": {}, "teacher2": {}}
        if teacher_row:
            for i in range(1, 14):
                teacher_scores["teacher1"][f"s{i}"] = teacher_row.get(f"score_s{i}_t1")
                teacher_scores["teacher2"][f"s{i}"] = teacher_row.get(f"score_s{i}_t2")

        # ✅ ส่งออกเป็น JSON
        return {
            "answer_id": answer["answer_id"],
            "student_id": answer["student_id"],
            "group_id": answer["group_id"],
            "exam_year": answer["exam_year"],
            "essay_text": answer["essay_text"],
            "essay_analysis": answer["essay_analysis"],
            "status": answer["status"],
            "score": answer["score"],
            "description": answer["description"],   # <- JSON ที่มี score/feedback ของ AI
            "teacher_scores": teacher_scores
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
