# 单图人物属性服务（离线构建）
#
# 远程宿主没有外网，pypi 与 docker registry 都不可达 → 镜像不能 pip install，
# 只能把宿主上**已经跑通的环境与权重**拷进来组装。
#
# 基础镜像与宿主同为 Ubuntu 22.04（22.04.5），glibc 版本一致；
# cv2 需要的 GL/X/glib 系统库基础镜像里没有，由 libs/ 补到 /usr/local/lib。
#
# 构建（在镜像目录下）：
#   docker build -t hc-attr:1.0 .
# 运行：
#   docker run -d --name hc-attr -p 18080:8080 --gpus all hc-attr:1.0
#   curl --data-binary @x.jpg http://127.0.0.1:18080/v1/analyze
FROM ubuntu:22.04

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    PATH=/opt/conda/envs/rfdetr/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONPATH=/opt/venv/lib/python3.11/site-packages \
    LD_LIBRARY_PATH=/opt/conda/envs/rfdetr/lib:/usr/local/lib \
    EG2_DIR=/opt/models/eg2 \
    HEADS=/opt/models/attr_heads.joblib \
    DETECTOR=/opt/models/yolo11m.pt \
    DEVICE=cuda:0

COPY libs/ /usr/local/lib/
COPY env/rfdetr /opt/conda/envs/rfdetr
COPY venv/site-packages /opt/venv/lib/python3.11/site-packages
COPY models/ /opt/models/
COPY app/ /opt/app/

# ★ 覆盖 PATH 时把 /sbin 排除了，ldconfig 在 /sbin 下 → 必须用绝对路径（否则 exit 127）
# ★ 不要加 `chmod -R /opt`：它会把 9 GB 环境整个重写一遍，导出阶段要多花十几分钟
RUN /sbin/ldconfig

WORKDIR /opt/app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz',timeout=3).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
