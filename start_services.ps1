# SearchEngine 服务启动脚本
# 自动启动 Docker Compose 和 ngrok

Write-Host "正在启动 SearchEngine 服务..." -ForegroundColor Green

# 检查是否在正确的目录
if (!(Test-Path "docker-compose.yml")) {
    Write-Host "错误: 请在项目根目录运行此脚本" -ForegroundColor Red
    exit 1
}

# 启动 Docker Compose 服务
Write-Host "启动 Docker 服务..." -ForegroundColor Yellow
Start-Process -FilePath "docker-compose" -ArgumentList "up", "--build" -NoNewWindow

# 等待服务启动
Write-Host "等待服务启动 (10秒)..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# 启动 ngrok (使用微软商店版本)
Write-Host "启动 ngrok 隧道..." -ForegroundColor Yellow
Start-Process -FilePath "ngrok" -ArgumentList "http", "5000" -NoNewWindow

Write-Host "服务启动完成!" -ForegroundColor Green
Write-Host "本地访问: http://localhost:5000" -ForegroundColor Cyan
Write-Host "请等待几秒钟，然后检查 ngrok 控制台获取公网地址" -ForegroundColor Cyan
Write-Host "ngrok 控制台: http://localhost:4040" -ForegroundColor Cyan

# 保持脚本运行
Write-Host "按任意键退出..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")