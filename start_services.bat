@echo off
chcp 65001 >nul
title SearchEngine 服务启动器

echo.
echo ========================================
echo    SearchEngine 服务启动器
echo ========================================
echo.

echo [1/3] 启动 Docker 服务...
start /B docker-compose up --build

echo [2/3] 等待服务启动...
timeout /t 10 /nobreak >nul

echo [3/3] 启动 ngrok 隧道...
start ngrok http 5000

echo.
echo ========================================
echo 服务启动完成！
echo ========================================
echo 本地访问: http://localhost:5000
echo ngrok 控制台: http://localhost:4040
echo.
echo 请等待几秒钟，然后访问 ngrok 控制台获取公网地址
echo.

pause