# Oracle Always Free Deployment (ECESSP)

This guide deploys ECESSP on a single Oracle Always Free VM with Docker Compose.

## 1) Provision VM

1. Create an OCI compute instance:
   - Shape: `VM.Standard.A1.Flex` (recommended)
   - OCPU/RAM: start with `2 OCPU / 12 GB`
   - OS: Ubuntu 22.04
2. Reserve a public IP and attach it to the VM.
3. In OCI Security List / NSG, allow inbound:
   - `22` (SSH)
   - `80` (HTTP)
   - `443` (HTTPS)
   - `5000` (temporary direct app access; optional)

## 2) Install Docker on VM

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release git
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

## 3) Deploy Project

```bash
git clone <your-repo-url> ECESSP
cd ECESSP
```

Create runtime env files:

```bash
cp ecessp-ml/.env.example ecessp-ml/.env
cp ecessp-frontend/.env.example ecessp-frontend/.env
```

Set strong secrets in:
- `ecessp-ml/.env` (`ECESSP_API_KEY`, optional bearer)
- `ecessp-frontend/.env` (`AUTH_USERNAME`, `AUTH_PASSWORD`)

Start services:

```bash
docker compose up -d --build
docker compose ps
```

## 4) Verify Health

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:5000/api/health
```

If these pass, app is live on VM IP:
- `http://<VM_PUBLIC_IP>:5000`

## 5) Domain + TLS (Caddy recommended)

Install Caddy:

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install -y caddy
```

Set DNS A record:
- `@` -> `<VM_PUBLIC_IP>`
- `www` -> `<VM_PUBLIC_IP>` (optional)

Configure `/etc/caddy/Caddyfile`:

```caddy
yourdomain.com, www.yourdomain.com {
    reverse_proxy 127.0.0.1:5000
    encode zstd gzip
}
```

Apply config:

```bash
sudo systemctl reload caddy
```

## 6) Ops Commands

```bash
docker compose logs -f ecessp-frontend
docker compose logs -f ecessp-ml
docker compose restart
docker compose pull
docker compose up -d --build
```

## 7) Production Notes

- Backend port `8000` is bound to `127.0.0.1` in compose (not public).
- Keep only `80/443` publicly reachable after TLS proxy is set.
- Always Free instances can be reclaimed if idle for long periods; keep periodic activity.
