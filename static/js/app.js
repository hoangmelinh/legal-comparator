function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

class App {
    constructor() {
        this.workflowId = `workflow_${Math.random().toString(36).slice(2, 11)}`;
        this.compareJobId = null;
        this.compareResult = null;
        this.currentView = "home";
        this.file1DocId = null;
        this.file2DocId = null;

        this.elements = {
            nav: document.querySelectorAll(".nav-item"),
            views: document.querySelectorAll(".view-section"),
            inputA: document.getElementById("input-a"),
            inputB: document.getElementById("input-b"),
            zoneA: document.getElementById("zone-a"),
            zoneB: document.getElementById("zone-b"),
            panelB: document.getElementById("panel-b"),
            btnCompare: document.getElementById("btn-compare"),
            loadingOverlay: document.getElementById("loading-overlay"),
            progressBar: document.getElementById("progress-bar"),
            progressPct: document.getElementById("progress-pct"),
            progressMsg: document.getElementById("progress-message"),
            progressTitle: document.getElementById("progress-title"),
            changesList: document.getElementById("changes-list"),
            changesCount: document.getElementById("changes-count"),
            btnViewReport: document.getElementById("btn-view-report"),
            tocList: document.getElementById("toc-list"),
            reportContent: document.getElementById("report-content"),
            btnAskChatbot: document.getElementById("btn-ask-chatbot"),
        };

        this.pdfA = new window.PdfRenderer("pdf-wrapper-a", "pdf-pages-a", "page-num-a", "page-count-a", {
            side: "old",
            scale: 1.12,
        });
        this.pdfB = new window.PdfRenderer("pdf-wrapper-b", "pdf-pages-b", "page-num-b", "page-count-b", {
            side: "new",
            scale: 1.12,
        });

        this.bindEvents();
        this.resetCompareArtifacts();
        this.resetFile2Ui();
    }

    bindEvents() {
        this.elements.nav.forEach((nav) => {
            nav.addEventListener("click", (event) => {
                event.preventDefault();
                if (nav.classList.contains("disabled")) return;
                this.switchView(nav.dataset.view);
            });
        });

        this.elements.zoneA.addEventListener("click", () => this.elements.inputA.click());
        this.elements.zoneB.addEventListener("click", () => {
            if (!this.elements.panelB.classList.contains("disabled")) {
                this.elements.inputB.click();
            }
        });
        this.elements.inputA.addEventListener("change", (event) => this.handleFileUpload(event.target.files[0], "file_1"));
        this.elements.inputB.addEventListener("change", (event) => this.handleFileUpload(event.target.files[0], "file_2"));
        this.elements.btnCompare.addEventListener("click", () => this.startCompare());
        this.elements.btnViewReport.addEventListener("click", () => this.switchView("report"));
        this.elements.btnAskChatbot.addEventListener("click", () => this.switchView("chat"));

        const preventDefaults = (event) => {
            event.preventDefault();
            event.stopPropagation();
        };

        ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
            this.elements.zoneA.addEventListener(eventName, preventDefaults, false);
            this.elements.zoneB.addEventListener(eventName, preventDefaults, false);
        });

        this.elements.zoneA.addEventListener("drop", (event) => this.handleFileUpload(event.dataTransfer.files[0], "file_1"));
        this.elements.zoneB.addEventListener("drop", (event) => {
            if (!this.elements.panelB.classList.contains("disabled")) {
                this.handleFileUpload(event.dataTransfer.files[0], "file_2");
            }
        });
    }

    switchView(viewId) {
        this.currentView = viewId;
        this.elements.nav.forEach((item) => item.classList.remove("active"));
        document.querySelector(`.nav-item[data-view="${viewId}"]`)?.classList.add("active");
        this.elements.views.forEach((view) => view.classList.remove("active"));
        document.getElementById(`view-${viewId}`)?.classList.add("active");
    }

    showProgress(title, message) {
        this.elements.loadingOverlay.classList.remove("hidden");
        this.elements.progressTitle.textContent = title;
        this.updateProgress(0, message);
    }

    updateProgress(percent, message) {
        const safePercent = Math.max(0, Math.min(100, Math.round(percent)));
        this.elements.progressBar.style.width = `${safePercent}%`;
        this.elements.progressPct.textContent = `${safePercent}%`;
        if (message) this.elements.progressMsg.textContent = message;
    }

    hideProgress() {
        this.elements.loadingOverlay.classList.add("hidden");
    }

    async pollProcessJob(jobId) {
        return new Promise((resolve, reject) => {
            const interval = setInterval(async () => {
                try {
                    const status = await window.api.getJobProgress(jobId);
                    this.updateProgress(status.progress_percent, status.message);
                    if (status.status === "completed") {
                        clearInterval(interval);
                        resolve(status);
                    } else if (status.status === "failed") {
                        clearInterval(interval);
                        reject(new Error(status.error || "Job failed"));
                    }
                } catch (error) {
                    clearInterval(interval);
                    reject(error);
                }
            }, 800);
        });
    }

    resetNavAfterCompare() {
        ["nav-compare", "nav-report", "nav-chat"].forEach((id) => {
            const el = document.getElementById(id);
            if (!el) return;
            el.classList.add("disabled");
            el.classList.remove("active");
        });
        document.querySelector('.nav-item[data-view="home"]')?.classList.add("active");
    }

    resetCompareArtifacts() {
        this.compareJobId = null;
        this.compareResult = null;
        this.elements.changesList.innerHTML = "";
        this.elements.changesCount.textContent = "0";
        this.elements.tocList.innerHTML = "";
        this.elements.reportContent.innerHTML = "";
        ["stat-total", "stat-added", "stat-removed", "stat-modified", "stat-minor", "stat-moved"].forEach((id) => {
            const el = document.getElementById(id);
            if (el) el.textContent = "0";
        });
        this.resetNavAfterCompare();
        this.pdfA.reset();
        this.pdfB.reset();
    }

    resetFile2Ui() {
        this.file2DocId = null;
        this.elements.panelB.classList.add("disabled");
        this.elements.btnCompare.setAttribute("disabled", "disabled");
        document.getElementById("h3-b").textContent = "Upload File 1 trước";
        document.getElementById("h3-b").classList.add("text-disabled");
        document.getElementById("p-b").textContent = "File 2 sẽ khả dụng sau khi chọn File 1";
        document.getElementById("p-b").classList.add("text-disabled");
        document.getElementById("dot-b").className = "dot gray";
        document.getElementById("status-b").textContent = "Chờ File 1";
        document.getElementById("status-b").className = "status-text text-disabled";
        document.getElementById("action-hint").textContent = "Vui lòng tải lên cả 2 file để tiếp tục";

        const stepNum = document.querySelector("#panel-b .step-num");
        if (stepNum) stepNum.classList.add("disabled-step");
        
        const iconB = document.querySelector("#zone-b i");
        if (iconB) {
            iconB.classList.remove("icon-default");
            iconB.classList.add("icon-disabled");
        }
    }

    unlockFile2Ui() {
        this.elements.panelB.classList.remove("disabled");
        
        const stepNum = document.querySelector("#panel-b .step-num");
        if (stepNum) stepNum.classList.remove("disabled-step");
        
        const iconB = document.querySelector("#zone-b i");
        if (iconB) {
            iconB.classList.remove("icon-disabled");
            iconB.classList.add("icon-default");
        }

        document.getElementById("h3-b").textContent = "Kéo thả hoặc nhấn để chọn";
        document.getElementById("h3-b").classList.remove("text-disabled");
        document.getElementById("p-b").textContent = "Hỗ trợ: .txt, .pdf, .docx";
        document.getElementById("p-b").classList.remove("text-disabled");
        document.getElementById("status-b").textContent = "Chưa tải lên";
        document.getElementById("status-b").className = "status-text text-gray";
        document.getElementById("action-hint").textContent = "Tải file 2 để sẵn sàng so sánh";
    }

    prepareForReplacingFile1() {
        this.elements.btnCompare.setAttribute("disabled", "disabled");
    }

    prepareForReplacingFile2() {
        this.elements.btnCompare.setAttribute("disabled", "disabled");
    }

    syncCompareButtonState() {
        if (this.file1DocId && this.file2DocId) {
            this.elements.btnCompare.removeAttribute("disabled");
            return;
        }
        this.elements.btnCompare.setAttribute("disabled", "disabled");
    }

    setProcessingState(slot, file) {
        const suffix = slot === "file_1" ? "a" : "b";
        const ext = (file.name.split(".").pop() || "").toUpperCase() || "FILE";
        document.getElementById(`h3-${suffix}`).textContent = file.name;
        document.getElementById(`p-${suffix}`).textContent = `${ext} • ${Math.round(file.size / 1024)} KB`;
        document.getElementById(`dot-${suffix}`).className = "dot gray";
        document.getElementById(`status-${suffix}`).textContent = "Đang xử lý";
        document.getElementById(`status-${suffix}`).className = "status-text text-gray";
    }

    setFailedState(slot, file, message) {
        const suffix = slot === "file_1" ? "a" : "b";
        const ext = (file.name.split(".").pop() || "").toUpperCase() || "FILE";
        document.getElementById(`h3-${suffix}`).textContent = file.name;
        document.getElementById(`p-${suffix}`).textContent = `${ext} • ${Math.round(file.size / 1024)} KB`;
        document.getElementById(`dot-${suffix}`).className = "dot red";
        document.getElementById(`status-${suffix}`).textContent = message || "Tải lên thất bại";
        document.getElementById(`status-${suffix}`).className = "status-text text-red";
    }

    async handleFileUpload(file, slot) {
        if (!file) return;
        const input = slot === "file_1" ? this.elements.inputA : this.elements.inputB;

        try {
            if (slot === "file_1") this.prepareForReplacingFile1();
            if (slot === "file_2") this.prepareForReplacingFile2();

            this.showProgress(`Đang tải file ${slot === "file_1" ? "1" : "2"}`, "Đang bắt đầu tải lên...");
            const uploadResult = await window.api.uploadDocument(file, slot, this.workflowId);

            if (slot === "file_1") {
                this.file1DocId = null;
                this.file2DocId = null;
                this.resetCompareArtifacts();
                this.resetFile2Ui();
                this.setProcessingState("file_1", file);
            } else {
                this.file2DocId = null;
                this.resetCompareArtifacts();
                this.setProcessingState("file_2", file);
            }

            const status = await this.pollProcessJob(uploadResult.job_id);
            this.hideProgress();

            if (slot === "file_1") {
                this.file1DocId = status.document_id || uploadResult.document_id;
                document.getElementById("h3-a").textContent = file.name;
                document.getElementById("p-a").textContent = `${file.name.split(".").pop().toUpperCase()} • ${Math.round(file.size / 1024)} KB`;
                document.getElementById("dot-a").className = "dot green";
                document.getElementById("status-a").textContent = "Đã xử lý xong";
                document.getElementById("status-a").className = "status-text text-green";
                this.unlockFile2Ui();
            } else {
                this.file2DocId = status.document_id || uploadResult.document_id;
                document.getElementById("h3-b").textContent = file.name;
                document.getElementById("p-b").textContent = `${file.name.split(".").pop().toUpperCase()} • ${Math.round(file.size / 1024)} KB`;
                document.getElementById("dot-b").className = "dot green";
                document.getElementById("status-b").textContent = "Đã xử lý xong";
                document.getElementById("status-b").className = "status-text text-green";
                document.getElementById("action-hint").textContent = "Sẵn sàng so sánh";
            }

            this.syncCompareButtonState();
            if (status.warning) console.warn(status.warning);
        } catch (error) {
            this.hideProgress();
            alert(`Lỗi upload: ${error.message}`);
            this.setFailedState(slot, file, "Tải lên thất bại");
            this.syncCompareButtonState();
        } finally {
            input.value = "";
        }
    }

    async startCompare() {
        if (!this.file1DocId || !this.file2DocId) {
            alert("Cần xử lý xong file 1 và file 2 trước khi so sánh.");
            return;
        }

        try {
            this.elements.btnCompare.setAttribute("disabled", "disabled");
            this.showProgress("Đang so sánh 2 file", "Đang khởi tạo thuật toán so sánh...");
            const response = await window.api.startCompare(this.workflowId);
            this.compareJobId = response.compare_job_id;
            await this.pollProcessJob(this.compareJobId);
            this.compareResult = await window.api.getCompareResult(this.compareJobId);
            this.hideProgress();
            this.populateCompareView();
            this.populateReportView();
            document.getElementById("nav-compare").classList.remove("disabled");
            document.getElementById("nav-report").classList.remove("disabled");
            document.getElementById("nav-chat").classList.remove("disabled");
            this.switchView("compare");
        } catch (error) {
            this.hideProgress();
            alert(`Lỗi so sánh: ${error.message}`);
        } finally {
            this.syncCompareButtonState();
        }
    }

    extractHighlights(changes, sourceKey) {
        const highlights = [];
        changes.forEach((change) => {
            const anchors = Array.isArray(change[sourceKey]) ? change[sourceKey] : [];

            anchors.forEach((anchor) => {
                const bbox = anchor?.bbox;
                const page = anchor?.page ?? anchor?.page_start;
                if (!Array.isArray(bbox) || bbox.length !== 4 || page === null || page === undefined || !Number.isFinite(Number(page))) {
                    return;
                }
                highlights.push({
                    change_id: change.id,
                    page_start: Number(page),
                    status: change.status,
                    mode: anchor.kind || change.highlight_mode || "inline",
                    strike: Boolean(anchor.strike),
                    x0: Number(bbox[0]),
                    y0: Number(bbox[1]),
                    x1: Number(bbox[2]),
                    y1: Number(bbox[3]),
                });
            });
        });
        return highlights;
    }

    getStatusLabel(status) {
        switch (status) {
            case "ADDED":
                return "Thêm mới";
            case "REMOVED":
                return "Đã xóa";
            case "MODIFIED":
                return "Đã sửa";
            case "MINOR-MODIFIED":
                return "Thay đổi nhỏ";
            case "MOVED":
                return "Di chuyển";
            default:
                return status || "Không rõ";
        }
    }

    populateCompareView() {
        if (!this.compareResult) return;
        const changes = Array.isArray(this.compareResult.changes) ? this.compareResult.changes : [];
        this.elements.changesCount.textContent = String(changes.length);

        const urlA = this.compareResult.document_a?.pdf_url || this.compareResult.document_a_pdf_url || window.api.getPdfUrl(this.compareResult.document_a?.document_id, this.workflowId);
        const urlB = this.compareResult.document_b?.pdf_url || this.compareResult.document_b_pdf_url || window.api.getPdfUrl(this.compareResult.document_b?.document_id, this.workflowId);
        const highlightsA = this.extractHighlights(changes, "highlight_anchors_a");
        const highlightsB = this.extractHighlights(changes, "highlight_anchors_b");

        this.pdfA.loadDocument(urlA, highlightsA);
        this.pdfB.loadDocument(urlB, highlightsB);

        this.elements.changesList.innerHTML = "";
        changes.forEach((change) => {
            const card = document.createElement("div");
            card.className = "change-card";
            card.dataset.changeId = change.id;
            card.innerHTML = `
                <div class="card-header">
                    <div class="card-article">${escapeHtml(change.article || "N/A")}</div>
                    <div class="badge ${escapeHtml(change.status)}">${escapeHtml(this.getStatusLabel(change.status))}</div>
                </div>
                <div class="card-summary">${escapeHtml(change.summary || change.change_type || "")}</div>
            `;
            card.addEventListener("click", () => this.focusChangeId(change.id));
            this.elements.changesList.appendChild(card);
        });
    }

    focusChangeId(id) {
        document.querySelectorAll(".change-card").forEach((card) => card.classList.remove("active"));
        const card = document.querySelector(`.change-card[data-change-id="${id}"]`);
        if (card) {
            card.classList.add("active");
            card.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }

        document.querySelectorAll(".hl-box").forEach((box) => box.classList.remove("active"));
        document.querySelectorAll(`.hl-box[data-change-id="${id}"]`).forEach((box) => box.classList.add("active"));

        const change = this.compareResult?.changes.find((item) => item.id === id);
        if (!change) return;
        const foundA = this.pdfA.goToChange(id);
        const foundB = this.pdfB.goToChange(id);
        if (!foundA && change.page_a) this.pdfA.goToPage(Number(change.page_a));
        if (!foundB && change.page_b) this.pdfB.goToPage(Number(change.page_b));
    }

    populateReportView() {
        if (!this.compareResult) return;
        const report = this.compareResult.report || {};
        const setStat = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };

        setStat("stat-total", report.total || 0);
        setStat("stat-added", report.added || 0);
        setStat("stat-removed", report.removed || 0);
        setStat("stat-modified", report.modified || 0);
        setStat("stat-minor", report.minor_modified || 0);
        setStat("stat-moved", report.moved || 0);

        this.elements.tocList.innerHTML = "";
        this.elements.reportContent.innerHTML = "";

        const reportChanges = Array.isArray(this.compareResult.report_changes) ? this.compareResult.report_changes : (Array.isArray(this.compareResult.changes) ? this.compareResult.changes : []);
        reportChanges.forEach((change) => {
            const statusLabel = this.getStatusLabel(change.status);
            const citationA = change.evidence?.citation_a || change.citation_a || change.raw?.citation_a || "";
            const citationB = change.evidence?.citation_b || change.citation_b || change.raw?.citation_b || "";

            const tocItem = document.createElement("div");
            tocItem.className = "toc-item";
            tocItem.innerHTML = `<span class="badge ${escapeHtml(change.status)}">${escapeHtml(statusLabel)}</span> ${escapeHtml(change.article || "N/A")}`;
            tocItem.addEventListener("click", () => document.getElementById(`report-item-${change.id}`)?.scrollIntoView({ behavior: "smooth" }));
            this.elements.tocList.appendChild(tocItem);

            const oldStyle = change.status === "ADDED" ? "opacity:0.5" : "";
            const newStyle = change.status === "REMOVED" ? "opacity:0.5" : "";

            const itemHtml = `
                <div class="report-detail-item" id="report-item-${escapeHtml(change.id)}">
                    <div class="report-detail-header">
                        <div style="font-weight: 600;">
                            ${escapeHtml(change.article || "N/A")}
                            <span class="badge ${escapeHtml(change.status)}" style="margin-left:8px;">${escapeHtml(statusLabel)}</span>
                        </div>
                        <div style="font-size:0.85rem; color: var(--text-muted);">${escapeHtml(change.change_type || "")}</div>
                    </div>
                    <div class="report-detail-body">
                        <p style="background: #f1f5f9; padding: 12px; border-radius: 6px; margin-bottom: 16px;">${escapeHtml(change.summary || "")}</p>
                        <div style="font-size: 0.9rem; font-weight: 600; color:var(--primary); margin-bottom: 8px;">Trích dẫn minh chứng</div>
                        <div style="display: flex; gap: 16px;">
                            <div style="flex:1; background: var(--c-del-bg); border: 1px solid #fecaca; padding: 12px; border-radius: 6px; font-size:0.9rem; ${oldStyle}">
                                <div style="font-weight: 600; color: var(--c-del); margin-bottom: 6px;">BẢN CŨ</div>
                                <div style="white-space: pre-wrap;">${escapeHtml(citationA || "Không có thay đổi/Đã thêm mới")}</div>
                            </div>
                            <div style="flex:1; background: var(--c-add-bg); border: 1px solid #bbf7d0; padding: 12px; border-radius: 6px; font-size:0.9rem; ${newStyle}">
                                <div style="font-weight: 600; color: var(--c-add); margin-bottom: 6px;">BẢN MỚI</div>
                                <div style="white-space: pre-wrap;">${escapeHtml(citationB || "Không có thay đổi/Đã xóa")}</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            this.elements.reportContent.insertAdjacentHTML("beforeend", itemHtml);
        });
    }
}

document.addEventListener("DOMContentLoaded", () => {
    window.app = new App();
});
