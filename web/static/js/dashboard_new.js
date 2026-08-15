/**
 * dashboard_new.js — Page logic for the redesigned dashboard (/dashboard_new).
 *
 * Starts as a copy of dashboard.js so /dashboard_new behaves identically to /
 * from the first commit; the redesign then lands here without touching the
 * live page. When it replaces index.html, this file becomes dashboard.js.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Model Selection
    document.querySelectorAll('.model-list-item').forEach(item => {
        item.addEventListener('click', async () => {
            const modelId = item.dataset.modelId;
            if (!modelId || item.classList.contains('active')) return;
            
            try {
                const res = await fetch(`/models/api/${modelId}/select`, { method: 'POST' });
                const data = await res.json();
                
                if (!res.ok) {
                    alert(data.error || 'Failed to select model');
                    return;
                }
                
                // Reload the page to show the new active model
                window.location.reload();
            } catch (err) {
                console.error(err);
                alert('Error selecting model');
            }
        });
    });

    // ---- Create / Edit Model Modal (shared) ----
    const createModal = document.getElementById('create-model-modal');
    const btnCreate = document.getElementById('btn-create-model');
    const btnCreateAlt = document.getElementById('btn-create-model-alt');
    const btnCancelCreate = document.getElementById('btn-cancel-create');
    const btnConfirmCreate = document.getElementById('btn-confirm-create');
    const formTitle = document.getElementById('model-form-title');
    const formWarning = document.getElementById('model-form-warning');
    const boardSizeSelect = document.getElementById('new-model-board-size');

    // null = create mode; otherwise holds the model being edited.
    let editingModel = null;

    const setField = (id, value) => {
        const el = document.getElementById(id);
        if (el != null && value !== undefined && value !== null) el.value = value;
    };

    // ---- Network size selector ----
    const netSlider = document.getElementById('new-model-net-size');
    const netLabelEl = document.getElementById('net-size-label');
    const netNoteEl = document.getElementById('net-size-note');
    const netParamsEl = document.getElementById('net-size-params');
    const netLockedEl = document.getElementById('net-size-locked');
    let netPresets = [];          // [{key,label,note,params,...}], smallest→largest
    let netDefaultKey = 'small';

    const fmtParams = (n) => {
        if (n >= 1e6) return (n / 1e6).toFixed(2).replace(/\.?0+$/, '') + 'M params';
        if (n >= 1e3) return Math.round(n / 1e3) + 'K params';
        return n + ' params';
    };

    const renderNetSize = () => {
        if (!netSlider || !netPresets.length) return;
        const idx = Math.max(0, Math.min(netPresets.length - 1, parseInt(netSlider.value) || 0));
        const p = netPresets[idx];
        if (netLabelEl) netLabelEl.textContent = p.label;
        if (netNoteEl) netNoteEl.textContent = p.note || '';
        if (netParamsEl) netParamsEl.textContent = fmtParams(p.params);
    };

    // Fetch presets (with live param counts) for a board size, keeping the
    // currently selected preset key when possible.
    const loadNetworkPresets = async (boardSize) => {
        if (!netSlider) return;
        const prevKey = netPresets.length
            ? (netPresets[parseInt(netSlider.value) || 0] || {}).key
            : null;
        try {
            const res = await fetch(`/models/api/network_presets?board_size=${boardSize}`);
            const data = await res.json();
            netPresets = data.presets || [];
            netDefaultKey = data.default || 'small';
        } catch (e) {
            console.error('Failed to load network presets', e);
            return;
        }
        netSlider.max = String(Math.max(0, netPresets.length - 1));
        const targetKey = prevKey || netDefaultKey;
        let idx = netPresets.findIndex(p => p.key === targetKey);
        if (idx < 0) idx = netPresets.findIndex(p => p.key === netDefaultKey);
        if (idx < 0) idx = 0;
        netSlider.value = String(idx);
        renderNetSize();
    };

    const setNetSizeToKey = (key) => {
        if (!netSlider || !netPresets.length) return;
        let idx = netPresets.findIndex(p => p.key === key);
        if (idx < 0) idx = netPresets.findIndex(p => p.key === netDefaultKey);
        if (idx < 0) idx = 0;
        netSlider.value = String(idx);
        renderNetSize();
    };

    const setNetSizeLocked = (locked) => {
        if (netSlider) netSlider.disabled = locked;
        if (netLockedEl) netLockedEl.style.display = locked ? '' : 'none';
    };

    if (netSlider) netSlider.addEventListener('input', renderNetSize);

    let modalParamBounds = null;

    const initModalParamSliders = async (values = {}) => {
        const container = document.getElementById('modal-param-categories');
        if (!container) return;
        if (!modalParamBounds) {
            modalParamBounds = await getParamBounds();
        }
        if (!modalParamBounds) return;

        container.innerHTML = buildParamSlidersHTML('modal-param', modalParamBounds, values);
        bindParamSliders('modal-param', modalParamBounds);
        setParamSliderValues('modal-param', modalParamBounds, values);
    };

    const closeCreateModal = () => { if (createModal) createModal.style.display = 'none'; };

    const openCreateModal = async () => {
        editingModel = null;
        if (formTitle) formTitle.textContent = 'Create New Model';
        btnConfirmCreate.textContent = 'Create Model';
        btnConfirmCreate.disabled = false;
        if (formWarning) formWarning.style.display = 'none';
        // Reset to defaults
        setField('new-model-name', '');
        setField('new-model-board-size', '9');
        setField('new-model-komi', '6.5');
        setField('new-model-ruleset', 'chinese');

        await initModalParamSliders();

        setNetSizeLocked(false);
        loadNetworkPresets(9).then(() => setNetSizeToKey(netDefaultKey));
        if (createModal) createModal.style.display = 'flex';
    };

    const openEditModal = async () => {
        try {
            const res = await fetch('/models/api/active');
            const model = await res.json();
            if (!model) { alert('No active model to edit'); return; }

            editingModel = model;
            if (formTitle) formTitle.textContent = `Edit: ${model.name}`;
            btnConfirmCreate.textContent = 'Save Changes';
            btnConfirmCreate.disabled = false;
            if (formWarning) formWarning.style.display = 'none';

            setField('new-model-name', model.name);
            setField('new-model-board-size', String(model.board_size));
            setField('new-model-komi', model.komi);
            setField('new-model-ruleset', model.ruleset);

            const t = model.training || {};
            await initModalParamSliders(t);

            // Network size is frozen after creation — show it, but locked.
            const netKey = (model.network && model.network.size_preset) || 'small';
            await loadNetworkPresets(model.board_size);
            setNetSizeToKey(netKey);
            setNetSizeLocked(true);

            // Expand advanced settings so all params are visible when editing.
            const adv = document.querySelector('.advanced-toggle');
            if (adv) adv.open = true;

            if (createModal) createModal.style.display = 'flex';
        } catch (e) {
            console.error(e);
            alert('Error loading model for editing');
        }
    };

    // Live warning when changing board size of a trained model (weights become incompatible).
    if (boardSizeSelect) {
        boardSizeSelect.addEventListener('change', () => {
            if (!formWarning) return;
            if (editingModel && editingModel.iteration > 0 &&
                parseInt(boardSizeSelect.value) !== editingModel.board_size) {
                formWarning.textContent = '⚠ Changing board size on a trained model discards its weights — training will restart from scratch.';
                formWarning.style.display = '';
            } else {
                formWarning.style.display = 'none';
            }
        });
        // Param counts depend on board size — refresh the preset readout.
        boardSizeSelect.addEventListener('change', () => {
            loadNetworkPresets(parseInt(boardSizeSelect.value) || 9);
        });
    }

    if (btnCreate) btnCreate.addEventListener('click', openCreateModal);
    if (btnCreateAlt) btnCreateAlt.addEventListener('click', openCreateModal);
    if (btnCancelCreate) btnCancelCreate.addEventListener('click', closeCreateModal);

    const btnEdit = document.getElementById('btn-edit');
    if (btnEdit) btnEdit.addEventListener('click', openEditModal);

    if (btnConfirmCreate) {
        btnConfirmCreate.addEventListener('click', async () => {
            const name = document.getElementById('new-model-name').value.trim();
            if (!name) {
                alert('Please enter a model name');
                return;
            }

            const paramValues = extractParamSliderValues('modal-param', modalParamBounds);

            const payload = {
                name: name,
                board_size: parseInt(document.getElementById('new-model-board-size').value),
                komi: parseFloat(document.getElementById('new-model-komi').value),
                ruleset: document.getElementById('new-model-ruleset').value,
                ...paramValues,
            };

            const isEdit = editingModel !== null;
            // Network size is only settable at creation time (frozen on edit).
            if (!isEdit && netPresets.length && netSlider) {
                const idx = Math.max(0, Math.min(netPresets.length - 1, parseInt(netSlider.value) || 0));
                payload.network_size = netPresets[idx].key;
            }
            const url = isEdit ? `/models/api/${editingModel.id}/update` : '/models/api/create';
            const busyText = isEdit ? 'Saving...' : 'Creating...';
            const restoreText = isEdit ? 'Save Changes' : 'Create Model';

            btnConfirmCreate.disabled = true;
            btnConfirmCreate.textContent = busyText;

            try {
                const res = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!res.ok) {
                    alert(data.error || 'Failed to save model');
                    btnConfirmCreate.disabled = false;
                    btnConfirmCreate.textContent = restoreText;
                    return;
                }
                if (data.warning) alert(data.warning);
                window.location.reload();
            } catch (err) {
                console.error(err);
                alert('Error saving model');
                btnConfirmCreate.disabled = false;
                btnConfirmCreate.textContent = restoreText;
            }
        });
    }

    // Copy Model
    const btnCopy = document.getElementById('btn-copy');
    if (btnCopy) {
        btnCopy.addEventListener('click', async () => {
            const activeItem = document.querySelector('.model-list-item.active');
            if (!activeItem) return;
            const modelId = activeItem.dataset.modelId;
            const currentName = document.getElementById('model-display-name').textContent;
            
            const newName = prompt('Enter name for the copied model:', currentName + ' Copy');
            if (!newName || newName.trim() === '') return;

            try {
                const res = await fetch(`/models/api/${modelId}/copy`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: newName })
                });
                if (res.ok) {
                    window.location.reload();
                } else {
                    const data = await res.json();
                    alert(data.error || 'Failed to copy');
                }
            } catch (e) {
                alert('Error copying model');
            }
        });
    }

    // Delete Model
    const btnDelete = document.getElementById('btn-delete');
    if (btnDelete) {
        btnDelete.addEventListener('click', async () => {
            const activeItem = document.querySelector('.model-list-item.active');
            if (!activeItem) return;
            const modelId = activeItem.dataset.modelId;
            const currentName = document.getElementById('model-display-name').textContent;
            
            if (!confirm(`Are you sure you want to delete the model "${currentName}"?\nThis cannot be undone and will delete all its training data and games.`)) {
                return;
            }

            try {
                const res = await fetch(`/models/api/${modelId}/delete`, {
                    method: 'DELETE'
                });
                if (res.ok) {
                    window.location.reload();
                } else {
                    const data = await res.json();
                    alert(data.error || 'Failed to delete');
                }
            } catch (e) {
                alert('Error deleting model');
            }
        });
    }
});
