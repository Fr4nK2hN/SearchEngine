# ngrok 固定网址说明

当前固定网址：

- <https://unadulating-otelia-dichotomously.ngrok-free.dev>

## 启动方式

在项目根目录运行：

```bash
bash tools/start_ngrok_searchengine.sh
```

这个脚本会做两件事：

1. 启动本地 `app` 服务
2. 使用固定 ngrok 域名把本地 `5000` 端口暴露到公网

## 前提

- 本机已经配置好 `ngrok authtoken`
- 本机已安装 `ngrok`
- Docker 服务正常运行

## 当前状态

我已经验证过下面这条命令可以正常复用固定地址：

```bash
ngrok http 5000 --url https://unadulating-otelia-dichotomously.ngrok-free.dev
```

## 说明

- 这个网址只有在 `ngrok` 进程运行时才能访问
- 关闭 `ngrok` 后，公网访问会失效
- 再次运行相同命令，可以继续使用同一个固定网址
