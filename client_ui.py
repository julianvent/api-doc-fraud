from __future__ import annotations

import requests
import streamlit as st

DEFAULT_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 600  # seconds — verify pipeline can be slow on first call

VERDICT_COLORS = {
    "ACCEPT": "green",
    "REVIEW": "orange",
    "REJECT": "red",
}


st.set_page_config(page_title="Fraud Detection Tester", layout="wide")
st.title("Fraud Detection Tester")


# Sidebar
with st.sidebar:
    st.header("Connection")
    base_url = st.text_input("API base URL", DEFAULT_BASE_URL).rstrip("/")

    if st.button("Ping /health", use_container_width=True):
        try:
            r = requests.get(f"{base_url}/v1/health", timeout=5)
            if r.ok:
                st.success(f"{r.status_code} — {r.text}")
            else:
                st.error(f"{r.status_code} — {r.text}")
        except requests.RequestException as exc:
            st.error(f"Connection failed: {exc}")


def render_meta(status_code: int, elapsed_ms: float) -> None:
    """Show HTTP status + latency next to a result."""
    color = "green" if 200 <= status_code < 300 else "red"
    st.markdown(
        f"**HTTP** :{color}[{status_code}] &nbsp;·&nbsp; **Latency** `{elapsed_ms:.0f} ms`"
    )


def post_multipart(url: str, files: list[tuple], data: dict) -> requests.Response | None:
    """POST multipart and surface connection errors inline instead of crashing."""
    try:
        return requests.post(url, files=files, data=data, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        return None


tab_verify, tab_template = st.tabs(["POST /v1/verify", "POST /v1/template"])

# --- /v1/verify ---->
with tab_verify:
    st.caption("Pipeline: metadata → tampering → preprocessor → ocr → policy")

    col_form, col_result = st.columns([1, 2], gap="large")

    with col_form:
        with st.form("verify_form", clear_on_submit=False):
            uploads = st.file_uploader(
                "document_images",
                accept_multiple_files=True,
                type=["png", "jpg", "jpeg", "pdf"],
                help="One or more document pages.",
            )
            doc_id = st.text_input("id", placeholder="e.g. test-001")
            doc_type = st.text_input("document_type (optional)", placeholder="e.g. passport")
            submit = st.form_submit_button("Send", type="primary", use_container_width=True)

    with col_result:
        if submit:
            if not uploads:
                st.warning("Upload at least one document image.")
            elif not doc_id.strip():
                st.warning("`id` is required.")
            else:
                files = [
                    ("document_images", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                    for f in uploads
                ]
                data = {"id": doc_id}
                if doc_type.strip():
                    data["document_type"] = doc_type

                with st.spinner("Calling /v1/verify…"):
                    resp = post_multipart(f"{base_url}/v1/verify", files=files, data=data)

                if resp is not None:
                    render_meta(resp.status_code, resp.elapsed.total_seconds() * 1000)

                    try:
                        body = resp.json()
                    except ValueError:
                        st.error("Response is not valid JSON.")
                        st.code(resp.text)
                    else:
                        if resp.ok and isinstance(body, dict):
                            verdict = str(body.get("verdict", "—"))
                            color = VERDICT_COLORS.get(verdict.upper(), "gray")
                            st.markdown(f"### Verdict: :{color}[**{verdict}**]")

                            m1, m2 = st.columns(2)
                            m1.metric("tampering_score", f"{body.get('tampering_score', 0):.4f}")
                            m2.metric("confidence", f"{body.get('confidence', 0):.4f}")

                            flags = body.get("flags") or []
                            st.markdown("**Flags**")
                            if flags:
                                st.markdown(" ".join(f"`{f}`" for f in flags))
                            else:
                                st.caption("_(none)_")

                            st.divider()

                        st.markdown("**Raw response**")
                        st.json(body, expanded=False)

# --- /v1/template ---->
with tab_template:
    st.caption("Uploads a reference image and extracts the field schema for that document type.")

    col_form, col_result = st.columns([1, 2], gap="large")

    with col_form:
        with st.form("template_form", clear_on_submit=False):
            img = st.file_uploader(
                "img",
                accept_multiple_files=False,
                type=["png", "jpg", "jpeg"],
                help="A single reference image for the template.",
            )
            tpl_doc_type = st.text_input("document_type", placeholder="e.g. passport")
            tpl_doc_name = st.text_input("document_name", placeholder="e.g. passport_us")
            tpl_country = st.text_input("country (optional)", placeholder="e.g. US")
            submit_tpl = st.form_submit_button("Send", type="primary", use_container_width=True)

    with col_result:
        if submit_tpl:
            if img is None:
                st.warning("Upload a template image.")
            elif not tpl_doc_type.strip() or not tpl_doc_name.strip():
                st.warning("`document_type` and `document_name` are required.")
            else:
                files = [("img", (img.name, img.getvalue(), img.type or "application/octet-stream"))]
                data = {"document_type": tpl_doc_type, "document_name": tpl_doc_name}
                if tpl_country.strip():
                    data["country"] = tpl_country

                with st.spinner("Calling /v1/template…"):
                    resp = post_multipart(f"{base_url}/v1/template", files=files, data=data)

                if resp is not None:
                    render_meta(resp.status_code, resp.elapsed.total_seconds() * 1000)

                    try:
                        body = resp.json()
                    except ValueError:
                        st.error("Response is not valid JSON.")
                        st.code(resp.text)
                    else:
                        if resp.ok and isinstance(body, dict):
                            m1, m2, m3 = st.columns(3)
                            m1.metric("document_name", body.get("document_name", "—"))
                            m2.metric("document_type", body.get("document_type", "—"))
                            m3.metric("country", body.get("country") or "—")

                            st.markdown(f"**img_path** `{body.get('img_path', '—')}`")

                            fields = body.get("fields") or {}
                            if fields:
                                st.markdown("**Field groups**")
                                cols = st.columns(max(len(fields), 1))
                                for col, (group, items) in zip(cols, fields.items()):
                                    col.metric(group, len(items or []))

                            st.divider()

                        st.markdown("**Raw response**")
                        st.json(body, expanded=False)
