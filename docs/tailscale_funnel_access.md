# Tailscale Funnel 访问说明

当前项目已经通过 Tailscale Funnel 对外暴露，公网地址是：

```text
https://search-demo.tail53faab.ts.net/
```

## 当前映射关系

```text
https://search-demo.tail53faab.ts.net/
  -> Tailscale Funnel
  -> http://127.0.0.1:5000
  -> Flask / Gunicorn
```

## 已验证结果

- 公网 `HEAD /` 返回 `HTTP/2 200`
- 页面首页 HTML 已经可以正常获取
- TLS 证书已经签发成功，证书域名为 `search-demo.tail53faab.ts.net`

## 这次排查到的关键点

Tailscale Funnel 在 userspace 模式下，不能只传 `--state`，还需要传 `--statedir`。

如果没有 `--statedir`，HTTPS 会在握手阶段失败，日志里会出现：

```text
no TailscaleVarRoot
```

原因是 Funnel 需要本地目录来保存 HTTPS 证书和相关运行时文件。

## 当前配置

当前运行时等价于：

```bash
tailscaled \
  --verbose=1 \
  --tun=userspace-networking \
  --socket=/Users/frank/Develop/SearchEngine/.tailscale/tailscaled.sock \
  --state=/Users/frank/Develop/SearchEngine/.tailscale/tailscaled.state \
  --statedir=/Users/frank/Develop/SearchEngine/.tailscale
```

然后启用：

```bash
tailscale --socket=/Users/frank/Develop/SearchEngine/.tailscale/tailscaled.sock funnel --bg 5000
```

## 注意

- 只要本地 `tailscaled` 进程停止，公网访问就会失效
- 只要本地 `app` 容器停止，公网访问也会失效
- 这个地址通常是稳定的，但前提是节点主机名和 tailnet 不变
- 如果以后再次修改 Tailscale hostname，公网地址也会跟着变化
