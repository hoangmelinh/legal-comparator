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
}

const api = new ApiClient();
window.api = api;
