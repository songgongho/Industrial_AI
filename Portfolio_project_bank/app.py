# -*- coding: utf-8 -*-
"""
Created on Tue May 20 19:12:12 2025
@author: User
"""

from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'secretkey123'  # 세션 유지용 키

# 사용자 정보 (아이디, 비밀번호, 잔액)
users = {
    "user1": {"password": "1234", "balance": 10000},
    "user2": {"password": "abcd", "balance": 5000}
}

# 메인 페이지 → 로그인으로 이동
@app.route('/')
def home():
    return redirect(url_for('login'))

# 로그인 처리
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uid = request.form['username']
        pwd = request.form['password']
        if uid in users and users[uid]['password'] == pwd:
            session['user'] = uid
            return redirect(url_for('dashboard'))
        return "❌ 로그인 실패: 아이디 또는 비밀번호가 잘못되었습니다."
    return render_template('login.html')

# 대시보드 (잔액 확인 및 입출금 처리)
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    uid = session['user']
    balance = users[uid]['balance']

    if request.method == 'POST':
        try:
            amount = int(request.form['amount'])
            if amount < 0:
                return "❌ 금액은 음수가 될 수 없습니다."
        except ValueError:
            return "❌ 유효한 숫자를 입력하세요."

        if 'deposit' in request.form:
            users[uid]['balance'] += amount
        elif 'withdraw' in request.form:
            if users[uid]['balance'] >= amount:
                users[uid]['balance'] -= amount
            else:
                return f"❌ 출금 실패: 잔액 부족 (현재 잔액: {balance}원)"
        return redirect(url_for('dashboard'))

    return render_template('dashboard.html', user=uid, balance=balance)

# 로그아웃
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# 서버 실행
if __name__ == '__main__':
    app.run(debug=True)
