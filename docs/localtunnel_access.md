# LocalTunnel 访问说明

当前项目已经安装了 `localtunnel`，可以用下面的命令启动一个临时公网地址：

```bash
bash tools/start_localtunnel_searchengine.sh
```

脚本会先确保 `app` 容器启动，再把本机 `127.0.0.1:5000` 暴露出去。

## 这次排查的实际结果

我已经确认：

- 宿主机访问 `http://127.0.0.1:5000` 返回 `200`
- 容器内访问 `http://127.0.0.1:5000` 返回 `200`
- `localtunnel` 可以成功分配公网地址

但我从公网回测拿到的是：

```text
HTTP/1.1 503 Service Unavailable
x-localtunnel-status: Tunnel Unavailable
```

这说明：

- 你的本地应用是正常的
- `localtunnel` 客户端也已经启动
- 当前失败点在公开的 `localtunnel` 中继服务，不在本地 Flask 应用

## 建议

如果你只是临时测试，可以重新运行脚本多试几次，使用随机域名。

如果你要给老师或同学稳定访问，不建议依赖公开的 `localtunnel` relay。更稳妥的做法是：

- 使用 Cloudflare Tunnel
- 或者把项目部署到一台云服务器
- 或者自建 LocalTunnel server
