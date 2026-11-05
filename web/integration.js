(() => {
  const mode = $('mode'), select = $('model'), preview = $('preview'), run = $('run');
  const demoOptions = select.innerHTML;
  const result = $('result'), empty = $('empty'), notice = document.querySelector('.notice');
  const evaluation = $('evaluation'), compare = $('compare');
  const demoEval = '<div class="card"><h2>Hiệu suất trên tập Test</h2><p>Chưa có kết quả đánh giá trong chế độ minh họa. Chuyển sang Mô hình thật để đọc metrics.json.</p></div>';
  evaluation.innerHTML = demoEval;
  const demoCompare = compare.innerHTML, demoNotice = notice.textContent;
  const demoResult = result.innerHTML, demoEmpty = empty.innerHTML;
  const demoRun = run.onclick, demoDownload = $('download').onclick;
  const demoCompareBody = $('comparebody').innerHTML;
  const stats = document.querySelectorAll('.stat');
  const demoStats = [stats[1].innerHTML, stats[3].innerHTML];
  let models = [], generation = 0, scan = 0, assessment = 0, source = null, last = null;
  const info = document.createElement('p');
  info.className = 'subtle'; info.setAttribute('role', 'status');
  run.after(info);
  const esc = value => String(value).replace(/[&<>"']/g, ch =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[ch]);
  const pct = p => new Intl.NumberFormat('vi-VN', {style:'percent', maximumFractionDigits:2}).format(p);
  const real = () => mode.value === 'real';
  const ready = () => models.filter(m => m.ready);
  const sample = () => preview.getAttribute('src')?.startsWith('data:');
  const imageReady = () => !preview.hidden && !!preview.getAttribute('src') && !sample();
  function sync() {
    if (!real()) return;
    const n = ready().length;
    run.disabled = !imageReady() || !select.value;
    const summary = n ? n + ' mô hình đã có file và cấu hình. Chọn ảnh JPG/PNG để chạy.' :
      'Chưa có mô hình sẵn sàng. Thêm model.pth hoặc model.onnx và model_config.json vào models/.';
    const issues = models.filter(m => !m.ready).map(m => m.name + ': ' + m.error);
    info.textContent = [summary, ...issues].join(' · ');
  }
  async function request(path, init) {
    const response = await fetch(path, init);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Yêu cầu không thành công.');
    return data;
  }
  function options(list) {
    return list.map(m => '<option value="' + esc(m.id) + '">' + esc(m.name) + '</option>').join('');
  }
  function resetOutput() {
    ++scan; last = null; result.hidden = true; empty.hidden = false;
  }
  function renderEvaluation() {
    evaluation.innerHTML = '<div class="card"><div class="cardhead"><h2>Đánh giá trên tập Test</h2><select id="metrics-model" aria-label="Mô hình đánh giá">' +
      options(models) + '</select></div><div id="metrics-output" role="status"></div>' +
      '<h3>So sánh chỉ số</h3><div id="metrics-overview" class="tablewrap"></div></div>';
    const picker = $('metrics-model');
    async function update() {
      const ticket = ++assessment, target = $('metrics-output');
      if (!picker.value) { target.textContent = 'Chưa có thư mục mô hình để đọc kết quả đánh giá.'; return; }
      target.textContent = 'Đang đọc kết quả đánh giá…';
      try {
        const d = await request('/api/metrics/' + encodeURIComponent(picker.value));
        if (ticket !== assessment || !real()) return;
        target.innerHTML = '<p>Tập Test: <b>' + esc(d.test_set_id) + '</b> · ' + d.samples + ' ảnh</p>' +
          '<div class="tablewrap"><table><tbody>' + Object.entries(d.scores).map(([k,v]) =>
            '<tr><th>' + esc(k) + '</th><td>' + pct(v) + '</td></tr>').join('') +
          '</tbody></table></div><h3>Confusion Matrix</h3><p>Hàng: nhãn thực · Cột: nhãn dự đoán</p>' +
          '<div class="tablewrap"><table><thead><tr><th>Thực / dự đoán</th>' +
          d.labels.map(l => '<th>' + esc(l) + '</th>').join('') + '</tr></thead><tbody>' +
          d.confusion_matrix.map((row,i) => '<tr><th>' + esc(d.labels[i]) + '</th>' +
            row.map((n,j) => '<td' + (i===j ? ' style="background:#e1f3ef;font-weight:bold"' : '') + '>' + n + '</td>').join('') +
            '</tr>').join('') + '</tbody></table></div>';
      } catch (e) { if (ticket === assessment && real()) target.textContent = e.message; }
    }
    picker.onchange = update; update();
    const summary = $('metrics-overview');
    request('/api/metrics').then(({models: rows}) => {
      if (!rows.length) { summary.textContent = 'Chưa có dữ liệu đánh giá.'; return; }
      const fields = ['Accuracy', 'Precision Macro', 'Recall Macro', 'F1-Score Macro'];
      summary.innerHTML = '<table><thead><tr><th>Mô hình</th><th>Tập Test</th>' +
        fields.map(x => '<th>' + esc(x) + '</th>').join('') + '</tr></thead><tbody>' +
        rows.map(r => '<tr><td>' + esc(r.model) + '</td><td>' +
          esc(r.available ? r.test_set_id : 'Chưa có dữ liệu') + '</td>' +
          fields.map(f => '<td>' + (r.available ? pct(r.scores[f]) : '—') + '</td>').join('') +
          '</tr>').join('') + '</tbody></table><p>Chỉ đối chiếu trực tiếp nếu dùng cùng tập Test.</p>';
    }).catch(e => { summary.textContent = e.message; });
  }
  function renderCompare() {
    compare.innerHTML = '<div class="card"><div class="cardhead"><h2>Đối chiếu trên cùng một ảnh</h2>' +
      '<button id="compare-run" class="btn primary">Chạy so sánh</button></div>' +
      '<p>Chọn ảnh thật ở mục Phân tích ảnh MRI trước khi so sánh.</p><div id="compare-output" class="tablewrap" role="status"></div></div>';
    $('compare-run').onclick = async () => {
      const target = $('compare-output'), src = preview.getAttribute('src');
      if (!imageReady() || !ready().length) { target.textContent = 'Cần một ảnh JPG/PNG và ít nhất một mô hình sẵn sàng.'; return; }
      const g = generation, button = $('compare-run');
      target.textContent = 'Đang chạy các mô hình…'; button.disabled = true;
      try {
        const blob = await (await fetch(src)).blob();
        const {results} = await request('/api/compare', {method:'POST', body:blob});
        const rows = results.map(d => d.error ? [d.model, '—', '—', d.error] :
          [d.model, d.ranked[0].label, pct(d.ranked[0].probability), d.inference_ms + ' ms']);
        if (real() && generation === g && src === preview.getAttribute('src')) {
          target.innerHTML = '<table><thead><tr><th>MÔ HÌNH</th><th>TOP-1</th><th>ĐIỂM</th><th>SUY LUẬN / TRẠNG THÁI</th></tr></thead><tbody>' +
            rows.map(r => '<tr>' + r.map(v => '<td>' + esc(v) + '</td>').join('') + '</tr>').join('') + '</tbody></table>';
        }
      } catch (e) { target.textContent = e.message; }
      finally { button.disabled = false; if (target.textContent === 'Đang chạy các mô hình…') target.textContent = ''; }
    };
  }
  function saveLast() {
    if (!last) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(last,null,2)], {type:'application/json'}));
    const a = document.createElement('a'); a.href = url; a.download = 'neuroscan-prediction.json'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function predictReal() {
    if (!imageReady() || !select.value) return;
    const id = select.value, src = preview.getAttribute('src'), ticket = ++scan;
    run.disabled = true; last = null; result.hidden = true; empty.hidden = false;
    info.textContent = 'Đang chạy mô hình…'; $('err').textContent = '';
    try {
      const blob = await (await fetch(src)).blob();
      const d = await request('/api/predict/' + encodeURIComponent(id), {method:'POST',body:blob});
      if (ticket !== scan || !real() || select.value !== id || src !== preview.getAttribute('src')) return;
      last = d;
      result.innerHTML = '<div class="resultlabel">DỰ ĐOÁN MÔ HÌNH · ' + esc(d.model) +
        '</div><div class="diagnosis"><div><small>Nhãn dự đoán</small><strong>' + esc(d.ranked[0].label) +
        '</strong></div><div class="score">' + pct(d.ranked[0].probability) +
        '</div></div><p>Thời gian suy luận: ' + esc(d.inference_ms) + ' ms</p>' +
        '<h3>Xác suất 4 lớp · Top-3 ở ba hàng đầu</h3>' + d.ranked.map((r,i) =>
          '<div class="rank"><div class="rankline"><span>' + (i+1) + '. ' + esc(r.label) +
          '</span><b>' + pct(r.probability) + '</b></div><div class="track"><div class="bar" style="width:' +
          (r.probability*100) + '%"></div></div></div>').join('') +
        '<p class="subtle">Điểm do mô hình trả về. Chỉ dùng cho đồ án nghiên cứu.</p>' +
        '<button class="btn" id="real-download" style="margin-top:12px">↓ Tải kết quả</button>';
      $('real-download').onclick = saveLast; empty.hidden = true; result.hidden = false;
    } catch (e) { if (ticket === scan && real()) $('err').textContent = e.message; }
    finally { if (ticket === scan && real()) sync(); }
  }
  mode.onchange = async () => {
    const g = ++generation; ++assessment; resetOutput(); $('err').textContent = '';
    $('sample').hidden = real();
    if (!real()) {
      select.innerHTML = demoOptions; select.disabled = false;
      notice.textContent = demoNotice; info.textContent = '';
      stats[1].innerHTML = demoStats[0]; stats[3].innerHTML = demoStats[1];
      result.innerHTML = demoResult; empty.innerHTML = demoEmpty;
      $('download').onclick = demoDownload;
      compare.innerHTML = demoCompare; $('comparebody').innerHTML = demoCompareBody;
      evaluation.innerHTML = demoEval;
      run.textContent = 'Chạy minh họa kết quả →'; run.disabled = preview.hidden;
      run.onclick = demoRun; refreshPreview(); return;
    }
    empty.innerHTML = '<div><div class="orbit">◎</div><b>Sẵn sàng dự đoán</b><p>Chọn ảnh và mô hình, sau đó chạy dự đoán.</p></div>';
    notice.textContent = 'Chế độ mô hình thật · Ảnh được gửi đến tiến trình NeuroScan chạy trên máy này để suy luận. Chỉ dùng ảnh được phép sử dụng và không dùng để chẩn đoán.';
    if (sample()) $('reset').click();
    select.innerHTML = '<option value="">Đang kiểm tra…</option>'; select.disabled = true;
    run.textContent = 'Chạy dự đoán →'; run.onclick = predictReal; run.disabled = true;
    models = []; renderCompare(); renderEvaluation();
    try {
      const data = await request('/api/models');
      if (g !== generation) return;
      models = data.models;
      select.innerHTML = options(ready()) || '<option value="">Chưa có mô hình</option>';
      select.disabled = !ready().length;
      stats[1].innerHTML = '<small>Mô hình có file</small><strong>' + ready().length + '</strong><em>Đọc từ thư mục models/</em>';
      stats[3].innerHTML = '<small>Trạng thái hệ thống</small><strong style="font-size:20px">' +
        (ready().length ? 'Có mô hình' : 'Chờ mô hình') + '</strong><em>Chế độ dự đoán thật</em>';
      renderEvaluation(); sync(); refreshPreview();
    } catch (e) {
      if (g === generation) { select.innerHTML = '<option value="">Không kết nối được</option>'; info.textContent = 'Khởi chạy backend FastAPI theo README.md để dùng mô hình thật.'; }
    }
  };
  const resized = document.createElement('details');
  resized.hidden = true;
  resized.innerHTML = '<summary>Xem ảnh sau tiền xử lý</summary><p class="subtle">Ảnh sau đổi kích thước, trước chuẩn hóa tensor.</p><img alt="Ảnh sau đổi kích thước" style="max-width:224px;width:100%;border-radius:6px"><p role="status"></p>';
  preview.parentElement.after(resized);
  let resizedUrl, previewTicket = 0;
  async function refreshPreview() {
    const ticket = ++previewTicket, src = preview.getAttribute('src'), id = select.value;
    const img = resized.querySelector('img'), status = resized.querySelector('[role=status]');
    if (resizedUrl) URL.revokeObjectURL(resizedUrl);
    resizedUrl = null; img.removeAttribute('src'); img.hidden = true; status.textContent = '';
    resized.hidden = preview.hidden || (real() && (!id || !imageReady()));
    if (resized.hidden || !resized.open) return;
    try {
      if (!real()) {
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = 224;
        await preview.decode();
        if (ticket !== previewTicket) return;
        canvas.getContext('2d').drawImage(preview, 0, 0, 224, 224);
        img.src = canvas.toDataURL(); img.hidden = false; return;
      }
      const blob = await (await fetch(src)).blob();
      const reply = await fetch('/api/preview/' + encodeURIComponent(id), {method:'POST', body:blob});
      if (!reply.ok) throw new Error((await reply.json()).error);
      const data = await reply.blob();
      if (ticket !== previewTicket) return;
      resizedUrl = URL.createObjectURL(data); img.src = resizedUrl; img.hidden = false;
    } catch (e) { if (ticket === previewTicket) status.textContent = e.message; }
  }
  resized.ontoggle = refreshPreview;
  select.onchange = () => { resetOutput(); sync(); refreshPreview(); };
  new MutationObserver(() => {
    const next = preview.getAttribute('src');
    if (source === next) return;
    source = next; refreshPreview();
    if (real()) { resetOutput(); sync(); const o = $('compare-output'); if (o) o.textContent = ''; }
  }).observe(preview, {attributes:true, attributeFilter:['src','hidden']});
})();
