---
name: ci
description: Universal CI/CD pipeline generator for any tech stack — build, test, security scan, deploy. Supports GitHub Actions, GitLab CI, and Bitbucket Pipelines.
user-invocable: true
argument-hint: "<stack> [--platform github|gitlab|bitbucket] [--with-scan] [--with-deploy]"
---

# Universal CI/CD Pipeline Generator

Skill này cung cấp quy trình chuẩn hóa để tự động sinh file cấu hình **CI/CD Pipeline chất lượng cao, tối ưu tốc độ và tuân thủ các nguyên tắc bảo mật production** cho **mọi ngôn ngữ lập trình và nền tảng CI/CD**.

---

## 🎯 Cú pháp Kích hoạt (Usage)
```
/ci <stack> [--platform github|gitlab|bitbucket] [--with-scan] [--with-deploy]
```

### Các ví dụ đa dạng nền tảng:
- **Python / Data Science / Web:** `/ci Python (pytest, ruff) --platform github`
- **Node.js / Frontend / Fullstack:** `/ci Next.js / TypeScript --platform github --with-deploy`
- **Backend / Microservices:** `/ci Go / FastAPI / Java Spring / .NET --platform gitlab --with-scan --with-deploy`
- **Containerized Apps:** `/ci Docker Multi-service --platform bitbucket --with-scan`

---

## 🏗️ Các Giai đoạn Chuẩn của Pipeline (Pipeline Stages)

### 1. Build & Dependency Setup
- **Caching Dependencies:** Thiết lập cache package manager tương ứng (`pip`, `npm`/`pnpm`/`yarn`, `nuget`, `go mod`, `maven`/`gradle`, `cargo`) để tối ưu thời gian chạy.
- **Biên dịch & Đóng gói:** Compile mã nguồn, build production assets, hoặc build Docker image (multi-stage build, non-root user).

### 2. Test & Code Quality
- **Automated Tests:** Chạy Unit test, Integration test và xuất báo cáo độ phủ (Code Coverage: Cobertura, LCOV, JUnit XML).
- **Linting & Code Formatting:** Kiểm tra cú pháp và định dạng mã nguồn theo chuẩn của từng ngôn ngữ:
  - Python: `ruff`, `flake8`, `black`
  - JavaScript/TypeScript: `eslint`, `prettier`
  - Go: `golangci-lint`
  - C# / .NET: `dotnet format`
  - Java: `checkstyle`, `spotless`

### 3. Security Scan (Kích hoạt khi có `--with-scan`)
- **Dependency Audit:** Quét lỗ hổng bảo mật trong các thư viện phụ thuộc (`pip-audit`, `npm audit`, `trivy fs`, `dotnet list --vulnerable`).
- **Secret Scanning & SAST:** Kiểm tra rò rỉ secret hoặc quét mã nguồn tĩnh (Trivy, SonarQube, CodeQL).
- **Container Image Scanning:** Quét lỗ hổng bảo mật image với `trivy image` trước khi đẩy lên Registry.

### 4. Continuous Deployment (Kích hoạt khi có `--with-deploy`)
- **Registry Push:** Đẩy artifact/image lên Docker Hub, GitHub Container Registry (GHCR), AWS ECR, GCP Artifact Registry,...
- **Môi trường Thử nghiệm (Staging):** Tự động deploy khi merge vào nhánh `develop`/`staging`.
- **Môi trường Thực tế (Production):** Cấu hình cổng phê duyệt thủ công (**Manual Approval Gate** hoặc Environment Protection Rules).

---

## 📄 File Cấu hình Xuất ra theo Nền tảng
- **GitHub Actions:** `.github/workflows/ci.yml`
- **GitLab CI:** `.gitlab-ci.yml`
- **Bitbucket Pipelines:** `bitbucket-pipelines.yml`

---

## 🛡️ Các Nguyên tắc Vàng (Core Rules)
1. **Dependency Caching:** Bắt buộc áp dụng cache key theo hash của file dependencies (`requirements.txt`, `package-lock.json`, `go.sum`,...).
2. **Fail-Fast:** Bất kỳ bước test hoặc scan nghiêm trọng nào thất bại đều phải dừng pipeline ngay lập tức, không bao giờ deploy code lỗi.
3. **Zero Hardcoded Secrets:** Tuyệt đối không hardcode API keys, token, mật khẩu trong file pipeline; tất cả phải lấy từ Secret / Variable của nền tảng CI/CD.
4. **Isolated & Reproducible Environments:** Sử dụng container hoặc runner có phiên bản cụ thể (pinned OS & runtime versions).
