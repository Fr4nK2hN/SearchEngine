@echo off
chcp 65001 >nul
title SearchEngine 服务启动器

echo.
echo ========================================
echo    SearchEngine 服务启动器
echo ========================================
echo.

echo [1/2] 启动 Docker 服务...
start /B docker-compose up --build

echo [2/2] 等待服务启动...
timeout /t 10 /nobreak >nul

echo.
echo ========================================
echo 服务启动完成！
echo ========================================
echo 本地访问: http://localhost:5000
echo.

pause