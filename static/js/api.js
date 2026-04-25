class ApiClient {
    constructor(baseUrl = "/api") {
        this.baseUrl = baseUrl;
    }

    async _readErrorMessage(response, fallback) {
        try {
            const err = await response.json();
            return err.detail || err.message || fallback;
        } catch (jsonError) {
            try {
                const text = await response.text();
                return text || fallback;
            } catch (textError) {
                return fallback;
            }
        }
    }

    async uploadDocument(file, slot, workflowId = "default") {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(
            `${this.baseUrl}/documents/upload?slot=${encodeURIComponent(slot)}&workflow_id=${encodeURIComponent(workflowId)}`,
            {
                method: "POST",
                body: formData,
            }
        );

        if (!response.ok) {
            throw new Error(await this._readErrorMessage(response, "Upload failed"));
        }
        return await response.json();
    }

    async getJobProgress(jobId) {
        const response = await fetch(`${this.baseUrl}/progress/${encodeURIComponent(jobId)}`);
        if (!response.ok) {
            throw new Error(await this._readErrorMessage(response, "Failed to get progress"));
        }
        return await response.json();
    }

    async startCompare(workflowId = "default") {
        const response = await fetch(`${this.baseUrl}/compare`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ workflow_id: workflowId }),
        });
        if (!response.ok) {
            throw new Error(await this._readErrorMessage(response, "Compare start failed"));
        }
        return await response.json();
    }

    async getCompareProgress(compareJobId) {
        const response = await fetch(`${this.baseUrl}/compare/${encodeURIComponent(compareJobId)}/progress`);
        if (!response.ok) {
            throw new Error(await this._readErrorMessage(response, "Failed to get compare progress"));
        }
        return await response.json();
    }

    async getCompareResult(compareJobId) {
        const response = await fetch(`${this.baseUrl}/compare/${encodeURIComponent(compareJobId)}/result`);
        if (!response.ok) {
            throw new Error(await this._readErrorMessage(response, "Failed to get compare result"));
        }
        return await response.json();
    }

    getPdfUrl(documentId, workflowId = "default") {
        return `${this.baseUrl}/documents/${encodeURIComponent(documentId)}/pdf?workflow_id=${encodeURIComponent(workflowId)}`;
    }

    /**
     * Gửi tin nhắn đến chatbot và stream phản hồi token-by-token.
     * @param {string} workflowId
     * @param {string|null} compareJobId
     * @param {string} message
     * @param {function(string)} onToken - gọi mỗi khi có token mới
     * @param {function()} onDone - gọi khi stream kết thúc
     * @param {function(string)} onError - gọi khi có lỗi
     */
    async sendChatMessage(workflowId, compareJobId, message, history, onToken, onDone, onError) {
        try {
            const response = await fetch(`${this.baseUrl}/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    workflow_id: workflowId,
                    compare_job_id: compareJobId || null,
                    message: message,
                    history: history,
                }),
            });

            if (!response.ok) {
                const msg = await this._readErrorMessage(response, "Chat request failed");
                onError(msg);
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop(); // phần chưa hoàn chỉnh giữ lại

                for (const line of lines) {
                    if (!line.startsWith("data: ")) continue;
                    const payload = line.slice(6); // bỏ "data: "
                    if (payload === "[DONE]") {
                        onDone();
                        return;
                    }
                    // Khôi phục newlines đã escape
                    const text = payload.replace(/\\n/g, "\n");
                    onToken(text);
                }
            }
            onDone();
        } catch (err) {
            onError(err.message || "Network error");
        }
    }
}

const api = new ApiClient();
window.api = api;
