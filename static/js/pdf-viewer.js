class PdfRenderer {
    constructor(wrapperId, pagesId, pageNumId, pageCountId, options = {}) {
        this.wrapper = document.getElementById(wrapperId);
        this.pagesRoot = document.getElementById(pagesId);
        this.pageNumEl = document.getElementById(pageNumId);
        this.pageCountEl = document.getElementById(pageCountId);

        this.pdfDoc = null;
        this.currentHighlights = [];
        this.pageElements = new Map();
        this.renderToken = 0;
        this.scale = Number.isFinite(options.scale) ? options.scale : 1.12;
        this.side = options.side || "old";
        this.highlightFill = this.side === "new" ? "rgba(34, 197, 94, 0.18)" : "rgba(239, 68, 68, 0.18)";
        this.highlightBorder = this.side === "new" ? "rgba(34, 197, 94, 0.36)" : "rgba(239, 68, 68, 0.36)";

        if (this.wrapper) {
            this.wrapper.addEventListener("scroll", () => this.updateCurrentPage());
        }
    }

    reset() {
        this.pdfDoc = null;
        this.currentHighlights = [];
        this.pageElements.clear();
        this.renderToken += 1;
        if (this.pagesRoot) {
            this.pagesRoot.innerHTML = "";
        }
        if (this.pageNumEl) {
            this.pageNumEl.textContent = "0";
        }
        if (this.pageCountEl) {
            this.pageCountEl.textContent = "0";
        }
        if (this.wrapper) {
            this.wrapper.scrollTop = 0;
        }
    }

    normalizeHighlights(highlights) {
        const source = Array.isArray(highlights) ? highlights : [];
        const normalized = source
            .filter((item) => {
                const page = Number(item?.page_start);
                return Number.isFinite(page) && [item.x0, item.y0, item.x1, item.y1].every((value) => Number.isFinite(Number(value)));
            })
            .map((item) => ({
                ...item,
                page_start: Number(item.page_start),
                x0: Number(item.x0),
                y0: Number(item.y0),
                x1: Number(item.x1),
                y1: Number(item.y1),
            }))
            .sort((a, b) => (
                a.page_start - b.page_start ||
                String(a.change_id).localeCompare(String(b.change_id)) ||
                a.y0 - b.y0 ||
                a.x0 - b.x0
            ));

        const deduped = [];
        const seen = new Set();
        for (const item of normalized) {
            const key = [
                item.page_start,
                item.change_id,
                item.status,
                item.x0,
                item.y0,
                item.x1,
                item.y1,
            ].join("|");
            if (seen.has(key)) {
                continue;
            }
            seen.add(key);
            deduped.push(item);
        }
        return deduped;
    }

    async loadDocument(url, highlights = []) {
        this.currentHighlights = this.normalizeHighlights(highlights);
        this.renderToken += 1;
        const token = this.renderToken;

        try {
            this.pdfDoc = await pdfjsLib.getDocument(url).promise;
            if (token !== this.renderToken) {
                return;
            }

            if (this.pageCountEl) {
                this.pageCountEl.textContent = String(this.pdfDoc.numPages);
            }
            await this.renderAllPages({ preserveScroll: false, token });
        } catch (error) {
            console.error("Loi khi tai PDF:", error);
            this.reset();
        }
    }

    async renderAllPages({ preserveScroll = true, token = this.renderToken } = {}) {
        if (!this.pdfDoc || !this.pagesRoot || !this.wrapper) {
            return;
        }

        const previousRatio = preserveScroll && this.wrapper.scrollHeight > this.wrapper.clientHeight
            ? this.wrapper.scrollTop / Math.max(1, this.wrapper.scrollHeight - this.wrapper.clientHeight)
            : 0;

        this.pagesRoot.innerHTML = "";
        this.pageElements.clear();

        for (let pageNumber = 1; pageNumber <= this.pdfDoc.numPages; pageNumber += 1) {
            if (token !== this.renderToken) {
                return;
            }

            const page = await this.pdfDoc.getPage(pageNumber);
            const viewport = page.getViewport({ scale: this.scale });
            const outputScale = window.devicePixelRatio || 1;

            const pageNode = document.createElement("div");
            pageNode.className = "pdf-page";
            pageNode.dataset.page = String(pageNumber);

            const canvas = document.createElement("canvas");
            canvas.className = "pdf-page-canvas";
            canvas.width = Math.floor(viewport.width * outputScale);
            canvas.height = Math.floor(viewport.height * outputScale);
            canvas.style.width = `${viewport.width}px`;
            canvas.style.height = `${viewport.height}px`;

            const overlay = document.createElement("div");
            overlay.className = "page-highlights-layer";
            overlay.style.width = `${viewport.width}px`;
            overlay.style.height = `${viewport.height}px`;

            pageNode.appendChild(canvas);
            pageNode.appendChild(overlay);
            this.pagesRoot.appendChild(pageNode);
            this.pageElements.set(pageNumber, pageNode);

            const context = canvas.getContext("2d");
            await page.render({
                canvasContext: context,
                viewport,
                transform: outputScale !== 1 ? [outputScale, 0, 0, outputScale, 0, 0] : null,
            }).promise;

            this.drawHighlightsForPage(pageNumber, page, viewport, overlay);
        }

        if (preserveScroll) {
            this.wrapper.scrollTop = previousRatio * Math.max(0, this.wrapper.scrollHeight - this.wrapper.clientHeight);
        } else {
            this.wrapper.scrollTop = 0;
        }

        this.updateCurrentPage();
    }

    drawHighlightsForPage(pageNumber, page, viewport, overlay) {
        overlay.innerHTML = "";
        const pageHighlights = this.currentHighlights.filter((item) => item.page_start === pageNumber);
        if (!pageHighlights.length) {
            return;
        }

        const pageView = Array.isArray(page?.view) ? page.view : [0, 0, viewport.width, viewport.height];
        const sourceLeft = Number(pageView[0]) || 0;
        const sourceTop = Number(pageView[1]) || 0;
        const sourceWidth = Math.max(1, Math.abs(Number(pageView[2]) - Number(pageView[0]) || viewport.width));
        const sourceHeight = Math.max(1, Math.abs(Number(pageView[3]) - Number(pageView[1]) || viewport.height));
        const scaleX = viewport.width / sourceWidth;
        const scaleY = viewport.height / sourceHeight;

        pageHighlights.forEach((highlight) => {
            const x0 = Number(highlight.x0);
            const y0 = Number(highlight.y0);
            const x1 = Number(highlight.x1);
            const y1 = Number(highlight.y1);
            if (![x0, y0, x1, y1].every((value) => Number.isFinite(value))) {
                return;
            }

            const left = (Math.min(x0, x1) - sourceLeft) * scaleX;
            const top = (Math.min(y0, y1) - sourceTop) * scaleY;
            const width = Math.max(2, Math.abs(x1 - x0) * scaleX);
            const height = Math.max(2, Math.abs(y1 - y0) * scaleY);
            const clippedLeft = Math.max(0, Math.min(left, viewport.width - width));
            const clippedTop = Math.max(0, Math.min(top, viewport.height - height));

            const box = document.createElement("div");
            const mode = highlight.mode === "block" ? "block" : "inline";
            box.className = `hl-box hl-${mode}`;
            box.dataset.changeId = highlight.change_id;
            box.dataset.page = String(pageNumber);
            box.style.left = `${clippedLeft}px`;
            box.style.top = `${clippedTop}px`;
            box.style.width = `${width}px`;
            box.style.height = `${height}px`;
            box.style.backgroundColor = this.highlightFill;
            box.style.border = `1px solid ${this.highlightBorder}`;
            box.style.borderRadius = "2px";
            if (highlight.strike && this.side === "old") {
                const strike = document.createElement("span");
                strike.className = "hl-strike";
                box.appendChild(strike);
            }
            box.addEventListener("click", () => window.app.focusChangeId(highlight.change_id));
            overlay.appendChild(box);
        });
    }

    updateCurrentPage() {
        if (!this.pageElements.size || !this.pageNumEl || !this.wrapper) {
            if (this.pageNumEl) {
                this.pageNumEl.textContent = "0";
            }
            return;
        }

        const probe = this.wrapper.scrollTop + (this.wrapper.clientHeight * 0.2);
        let currentPage = 1;
        for (const [pageNumber, node] of this.pageElements.entries()) {
            if (node.offsetTop <= probe) {
                currentPage = pageNumber;
            } else {
                break;
            }
        }
        this.pageNumEl.textContent = String(currentPage);
    }

    goToPage(pageNumber) {
        const node = this.pageElements.get(Number(pageNumber));
        if (!node || !this.wrapper) {
            return;
        }
        this.wrapper.scrollTo({
            top: Math.max(0, node.offsetTop - 12),
            behavior: "smooth",
        });
    }

    goToChange(changeId) {
        const target = this.pagesRoot?.querySelector(`.hl-box[data-change-id="${changeId}"]`);
        if (!target) {
            return false;
        }
        const pageNode = target.closest(".pdf-page");
        if (!pageNode || !this.wrapper) {
            return false;
        }
        this.wrapper.scrollTo({
            top: Math.max(0, pageNode.offsetTop + target.offsetTop - 48),
            behavior: "smooth",
        });
        return true;
    }
}

window.PdfRenderer = PdfRenderer;
