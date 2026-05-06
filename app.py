"""
PDI Lab — Interface principal
Disciplina: Processamento Digital de Imagens
UFERSA / Departamento de Computação
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import os

from impl import C1, C2, C3

# ─────────────────────────────────────────────
#  PALETA E FONTES
# ─────────────────────────────────────────────
BG_DARK   = "#0d1117"
BG_PANEL  = "#161b22"
BG_CARD   = "#21262d"
BG_INPUT  = "#1c2128"
ACCENT    = "#58a6ff"
ACCENT2   = "#3fb950"
ACCENT3   = "#f78166"
TEXT_PRI  = "#e6edf3"
TEXT_SEC  = "#8b949e"
TEXT_DIM  = "#484f58"
BORDER    = "#30363d"

FONT_TITLE  = ("Courier New", 13, "bold")
FONT_LABEL  = ("Courier New", 9)
FONT_BTN    = ("Courier New", 9, "bold")
FONT_MONO   = ("Courier New", 8)
FONT_HEADER = ("Courier New", 11, "bold")


# ─────────────────────────────────────────────
#  UTILITÁRIOS DE UI
# ─────────────────────────────────────────────

def array_to_photoimage(arr: np.ndarray, max_w=380, max_h=300) -> ImageTk.PhotoImage:
    """Converte ndarray para PhotoImage, redimensionando para caber."""
    img = Image.fromarray(arr.astype(np.uint8))
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


def make_label(parent, text, fg=TEXT_SEC, font=FONT_LABEL, **kw):
    return tk.Label(parent, text=text, fg=fg, bg=BG_PANEL,
                    font=font, **kw)


def make_button(parent, text, command, color=ACCENT, **kw):
    btn = tk.Button(parent, text=text, command=command,
                    bg=BG_CARD, fg=color, activebackground=BG_INPUT,
                    activeforeground=TEXT_PRI, relief="flat",
                    font=FONT_BTN, cursor="hand2",
                    bd=0, padx=12, pady=6, **kw)
    btn.bind("<Enter>", lambda e: btn.config(bg=BG_INPUT))
    btn.bind("<Leave>", lambda e: btn.config(bg=BG_CARD))
    return btn


def make_entry(parent, width=18, **kw):
    return tk.Entry(parent, bg=BG_INPUT, fg=TEXT_PRI,
                    insertbackground=ACCENT, relief="flat",
                    font=FONT_MONO, width=width,
                    highlightthickness=1, highlightcolor=ACCENT,
                    highlightbackground=BORDER, **kw)


def make_combobox(parent, values, width=16):
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Dark.TCombobox",
                     fieldbackground=BG_INPUT, background=BG_CARD,
                     foreground=TEXT_PRI, arrowcolor=ACCENT,
                     bordercolor=BORDER, lightcolor=BORDER,
                     darkcolor=BORDER, selectbackground=BG_INPUT,
                     selectforeground=ACCENT)
    cb = ttk.Combobox(parent, values=values, width=width,
                      state="readonly", style="Dark.TCombobox",
                      font=FONT_MONO)
    cb.current(0)
    return cb


def section_title(parent, text):
    frm = tk.Frame(parent, bg=BG_PANEL)
    frm.pack(fill="x", pady=(14, 4))
    tk.Label(frm, text="▶ " + text, fg=ACCENT, bg=BG_PANEL,
             font=FONT_HEADER).pack(anchor="w")
    tk.Frame(frm, bg=BORDER, height=1).pack(fill="x", pady=(3, 0))


def image_card(parent, title="") -> tuple:
    """Retorna (frame, label_img, label_info)."""
    frm = tk.Frame(parent, bg=BG_CARD, bd=0,
                   highlightthickness=1, highlightbackground=BORDER)
    if title:
        tk.Label(frm, text=title, fg=TEXT_SEC, bg=BG_CARD,
                 font=FONT_MONO).pack(pady=(6, 0))
    lbl_img = tk.Label(frm, bg=BG_CARD)
    lbl_img.pack(padx=8, pady=6)
    lbl_info = tk.Label(frm, text="—", fg=TEXT_DIM, bg=BG_CARD, font=FONT_MONO)
    lbl_info.pack(pady=(0, 6))
    return frm, lbl_img, lbl_info


def show_array_in_label(arr: np.ndarray, lbl_img: tk.Label,
                         lbl_info: tk.Label, max_w=380, max_h=300):
    """Atualiza um label com a imagem do array."""
    ph = array_to_photoimage(arr, max_w, max_h)
    lbl_img.config(image=ph)
    lbl_img.image = ph   # manter referência
    h, w = arr.shape[:2]
    ch = arr.shape[2] if arr.ndim == 3 else 1
    lbl_info.config(text=f"{w}×{h} px | {ch}ch | dtype={arr.dtype}")


# ─────────────────────────────────────────────
#  WIDGET DE SELEÇÃO DE IMAGEM
# ─────────────────────────────────────────────

class ImageSelector(tk.Frame):
    """Barra compacta para selecionar imagem (botão + campo de texto)."""

    def __init__(self, parent, label="Imagem", on_load=None, **kw):
        super().__init__(parent, bg=BG_PANEL, **kw)
        self._on_load = on_load
        self._array = None

        tk.Label(self, text=label, fg=TEXT_SEC, bg=BG_PANEL,
                 font=FONT_LABEL, width=10, anchor="w").grid(row=0, column=0, padx=(0,4))

        self._entry = make_entry(self, width=34)
        self._entry.grid(row=0, column=1, padx=2)

        make_button(self, "Abrir…", self._browse, color=ACCENT
                    ).grid(row=0, column=2, padx=4)
        make_button(self, "Carregar", self._load_from_entry, color=ACCENT2
                    ).grid(row=0, column=3, padx=2)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Selecionar imagem",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.pgm *.ppm"),
                       ("Todos",   "*.*")])
        if path:
            self._entry.delete(0, tk.END)
            self._entry.insert(0, path)
            self._load_path(path)

    def _load_from_entry(self):
        path = self._entry.get().strip()
        if path:
            self._load_path(path)

    def _load_path(self, path):
        try:
            self._array = C1.load_image(path)
            if self._on_load:
                self._on_load(self._array, path)
        except Exception as e:
            messagebox.showerror("Erro ao carregar", str(e))

    @property
    def array(self):
        return self._array


# ─────────────────────────────────────────────
#  PÁGINA BASE
# ─────────────────────────────────────────────

class BasePage(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG_PANEL, **kw)

    def scrollable(self):
        """Envolve o frame em um canvas com scrollbar."""
        canvas = tk.Canvas(self, bg=BG_PANEL, bd=0,
                           highlightthickness=0)
        vsb = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=BG_PANEL)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))
        return inner


# ─────────────────────────────────────────────
#  PÁGINA C1 — Aritmética & Lógica
# ─────────────────────────────────────────────

class PageC1(BasePage):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        inner = self.scrollable()
        self._img1 = self._img2 = None
        self._build(inner)

    def _build(self, p):
        pad = dict(padx=20, pady=4, anchor="w")

        section_title(p, "Conjunto 1 — Operações Aritméticas e Lógicas")

        # Seletores de imagem
        sel_frm = tk.Frame(p, bg=BG_PANEL)
        sel_frm.pack(**pad, fill="x")
        self._sel1 = ImageSelector(sel_frm, "Imagem A",
                                   on_load=lambda a, _: setattr(self, '_img1', a))
        self._sel1.pack(pady=3, anchor="w")
        self._sel2 = ImageSelector(sel_frm, "Imagem B",
                                   on_load=lambda a, _: setattr(self, '_img2', a))
        self._sel2.pack(pady=3, anchor="w")

        # Operações Aritméticas
        section_title(p, "Aritméticas")
        arith_frm = tk.Frame(p, bg=BG_PANEL)
        arith_frm.pack(**pad)
        self._arith_op = make_combobox(arith_frm, ["add", "subtract", "multiply", "divide"])
        self._arith_op.grid(row=0, column=0, padx=(0, 10))
        make_button(arith_frm, "Executar", self._run_arith
                    ).grid(row=0, column=1)

        # Operações Lógicas
        section_title(p, "Lógicas")
        logic_frm = tk.Frame(p, bg=BG_PANEL)
        logic_frm.pack(**pad)
        self._logic_op = make_combobox(logic_frm, ["and", "or", "xor", "not"])
        self._logic_op.grid(row=0, column=0, padx=(0, 10))
        make_button(logic_frm, "Executar", self._run_logic
                    ).grid(row=0, column=1)

        # Cards de resultado
        section_title(p, "Resultado")
        cards_frm = tk.Frame(p, bg=BG_PANEL)
        cards_frm.pack(padx=20, pady=6, fill="x")

        frm1, self._lbl_img1, self._lbl_info1 = image_card(cards_frm, "Imagem A")
        frm1.grid(row=0, column=0, padx=6, pady=4, sticky="n")
        frm2, self._lbl_img2, self._lbl_info2 = image_card(cards_frm, "Imagem B")
        frm2.grid(row=0, column=1, padx=6, pady=4, sticky="n")
        frm3, self._lbl_res,  self._lbl_info3 = image_card(cards_frm, "Resultado")
        frm3.grid(row=0, column=2, padx=6, pady=4, sticky="n")

        make_button(p, "💾  Salvar resultado", self._save).pack(pady=8)

        self._result_arr = None

    def _validate(self, need_two=True):
        if self._img1 is None:
            messagebox.showwarning("Aviso", "Carregue a Imagem A primeiro.")
            return False
        if need_two and self._img2 is None:
            messagebox.showwarning("Aviso", "Carregue a Imagem B primeiro.")
            return False
        return True

    def _run_arith(self):
        if not self._validate():
            return
        try:
            op = self._arith_op.get()
            result = C1.aritmetica_imagens(self._img1, self._img2, op)
            self._show_result(result)
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _run_logic(self):
        op = self._logic_op.get()
        need_two = op != "not"
        if not self._validate(need_two):
            return
        try:
            result = C1.logica_imagens(self._img1, self._img2 if need_two else self._img1, op)
            self._show_result(result)
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _show_result(self, arr):
        self._result_arr = arr
        show_array_in_label(self._img1, self._lbl_img1, self._lbl_info1)
        if self._img2 is not None:
            show_array_in_label(self._img2, self._lbl_img2, self._lbl_info2)
        show_array_in_label(arr, self._lbl_res, self._lbl_info3)

    def _save(self):
        if self._result_arr is None:
            messagebox.showinfo("Info", "Nenhum resultado para salvar.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
               filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("BMP","*.bmp")])
        if path:
            C1.save_image(self._result_arr, path)
            messagebox.showinfo("Salvo", f"Imagem salva em:\n{path}")


# ─────────────────────────────────────────────
#  PÁGINA C2 — Transformações Geométricas
# ─────────────────────────────────────────────

class PageC2(BasePage):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        inner = self.scrollable()
        self._img = None
        self._result_arr = None
        self._compose_steps = []
        self._build(inner)

    def _build(self, p):
        pad = dict(padx=20, pady=4, anchor="w")
        section_title(p, "Conjunto 2 — Transformações Geométricas e Zoom")

        # Seletor
        self._sel = ImageSelector(p, "Imagem",
                                  on_load=self._on_img_load)
        self._sel.pack(**pad, fill="x")

        # ── Operações individuais ──
        ops_outer = tk.Frame(p, bg=BG_PANEL)
        ops_outer.pack(padx=20, pady=6, fill="x")

        # Translação
        section_title(ops_outer, "Translação")
        frm_t = tk.Frame(ops_outer, bg=BG_PANEL); frm_t.pack(anchor="w")
        tk.Label(frm_t, text="tx:", fg=TEXT_SEC, bg=BG_PANEL, font=FONT_LABEL).grid(row=0,column=0)
        self._tx = make_entry(frm_t, width=6); self._tx.insert(0,"50"); self._tx.grid(row=0,column=1,padx=4)
        tk.Label(frm_t, text="ty:", fg=TEXT_SEC, bg=BG_PANEL, font=FONT_LABEL).grid(row=0,column=2)
        self._ty = make_entry(frm_t, width=6); self._ty.insert(0,"30"); self._ty.grid(row=0,column=3,padx=4)
        make_button(frm_t, "Aplicar", lambda: self._apply("translacao",
            {"tx": int(self._tx.get()), "ty": int(self._ty.get())})).grid(row=0,column=4,padx=8)

        # Rotação
        section_title(ops_outer, "Rotação")
        frm_r = tk.Frame(ops_outer, bg=BG_PANEL); frm_r.pack(anchor="w")
        tk.Label(frm_r, text="ângulo (°):", fg=TEXT_SEC, bg=BG_PANEL, font=FONT_LABEL).grid(row=0,column=0)
        self._ang = make_entry(frm_r, width=6); self._ang.insert(0,"45"); self._ang.grid(row=0,column=1,padx=4)
        make_button(frm_r, "Aplicar", lambda: self._apply("rotacao",
            {"angulo": float(self._ang.get())})).grid(row=0,column=2,padx=8)

        # Escala
        section_title(ops_outer, "Escala")
        frm_e = tk.Frame(ops_outer, bg=BG_PANEL); frm_e.pack(anchor="w")
        tk.Label(frm_e, text="fx:", fg=TEXT_SEC, bg=BG_PANEL, font=FONT_LABEL).grid(row=0,column=0)
        self._fx = make_entry(frm_e, width=6); self._fx.insert(0,"1.5"); self._fx.grid(row=0,column=1,padx=4)
        tk.Label(frm_e, text="fy:", fg=TEXT_SEC, bg=BG_PANEL, font=FONT_LABEL).grid(row=0,column=2)
        self._fy = make_entry(frm_e, width=6); self._fy.insert(0,"1.5"); self._fy.grid(row=0,column=3,padx=4)
        make_button(frm_e, "Aplicar", lambda: self._apply("escala",
            {"fx": float(self._fx.get()), "fy": float(self._fy.get())})).grid(row=0,column=4,padx=8)

        # Reflexão
        section_title(ops_outer, "Reflexão")
        frm_ref = tk.Frame(ops_outer, bg=BG_PANEL); frm_ref.pack(anchor="w")
        self._eixo = make_combobox(frm_ref, ["horizontal","vertical","ambos"], width=14)
        self._eixo.grid(row=0,column=0,padx=(0,8))
        make_button(frm_ref, "Aplicar", lambda: self._apply("reflexao",
            {"eixo": self._eixo.get()})).grid(row=0,column=1)

        # Cisalhamento
        section_title(ops_outer, "Cisalhamento")
        frm_c = tk.Frame(ops_outer, bg=BG_PANEL); frm_c.pack(anchor="w")
        tk.Label(frm_c, text="shx:", fg=TEXT_SEC, bg=BG_PANEL, font=FONT_LABEL).grid(row=0,column=0)
        self._shx = make_entry(frm_c, width=6); self._shx.insert(0,"0.2"); self._shx.grid(row=0,column=1,padx=4)
        tk.Label(frm_c, text="shy:", fg=TEXT_SEC, bg=BG_PANEL, font=FONT_LABEL).grid(row=0,column=2)
        self._shy = make_entry(frm_c, width=6); self._shy.insert(0,"0.0"); self._shy.grid(row=0,column=3,padx=4)
        make_button(frm_c, "Aplicar", lambda: self._apply("cisalhamento",
            {"shx": float(self._shx.get()), "shy": float(self._shy.get())})).grid(row=0,column=4,padx=8)

        # ── Zoom ──
        section_title(ops_outer, "Zoom IN")
        frm_zi = tk.Frame(ops_outer, bg=BG_PANEL); frm_zi.pack(anchor="w")
        self._zoom_in_tipo = make_combobox(frm_zi, ["replicacao","interpolacao"], width=14)
        self._zoom_in_tipo.grid(row=0,column=0,padx=(0,6))
        tk.Label(frm_zi, text="fator:", fg=TEXT_SEC, bg=BG_PANEL, font=FONT_LABEL).grid(row=0,column=1)
        self._zoom_in_f = make_entry(frm_zi, width=6); self._zoom_in_f.insert(0,"2"); self._zoom_in_f.grid(row=0,column=2,padx=4)
        make_button(frm_zi, "Aplicar", self._run_zoom_in).grid(row=0,column=3,padx=8)

        section_title(ops_outer, "Zoom OUT")
        frm_zo = tk.Frame(ops_outer, bg=BG_PANEL); frm_zo.pack(anchor="w")
        self._zoom_out_tipo = make_combobox(frm_zo, ["exclusao","valor_medio"], width=14)
        self._zoom_out_tipo.grid(row=0,column=0,padx=(0,6))
        tk.Label(frm_zo, text="fator:", fg=TEXT_SEC, bg=BG_PANEL, font=FONT_LABEL).grid(row=0,column=1)
        self._zoom_out_f = make_entry(frm_zo, width=6); self._zoom_out_f.insert(0,"2"); self._zoom_out_f.grid(row=0,column=2,padx=4)
        make_button(frm_zo, "Aplicar", self._run_zoom_out).grid(row=0,column=3,padx=8)

        # ── Compostas ──
        section_title(p, "Transformação Composta")
        comp_outer = tk.Frame(p, bg=BG_PANEL)
        comp_outer.pack(padx=20, pady=4, fill="x")

        tk.Label(comp_outer, text="Adicione etapas na ordem desejada:",
                 fg=TEXT_SEC, bg=BG_PANEL, font=FONT_LABEL).pack(anchor="w")

        step_frm = tk.Frame(comp_outer, bg=BG_PANEL); step_frm.pack(anchor="w", pady=4)
        self._step_op = make_combobox(step_frm, C2.operacoes_disponiveis(), width=20)
        self._step_op.grid(row=0,column=0,padx=(0,8))
        tk.Label(step_frm, text="params (JSON):", fg=TEXT_SEC, bg=BG_PANEL, font=FONT_LABEL).grid(row=0,column=1)
        self._step_params = make_entry(step_frm, width=28)
        self._step_params.insert(0,'{"angulo": 45}')
        self._step_params.grid(row=0,column=2,padx=4)
        make_button(step_frm, "+ Etapa", self._add_step, color=ACCENT2).grid(row=0,column=3,padx=4)

        self._steps_list = tk.Listbox(comp_outer, bg=BG_INPUT, fg=TEXT_PRI,
                                       font=FONT_MONO, height=5, width=60,
                                       selectbackground=BG_CARD, bd=0,
                                       highlightthickness=1, highlightcolor=BORDER)
        self._steps_list.pack(pady=4)

        btn_row = tk.Frame(comp_outer, bg=BG_PANEL); btn_row.pack(anchor="w")
        make_button(btn_row, "Remover selecionada", self._remove_step, color=ACCENT3).pack(side="left", padx=4)
        make_button(btn_row, "Limpar tudo", self._clear_steps, color=TEXT_DIM).pack(side="left", padx=4)
        make_button(btn_row, "▶ Executar composição", self._run_compose, color=ACCENT2).pack(side="left", padx=12)

        # ── Visualização ──
        section_title(p, "Visualização")
        vis_frm = tk.Frame(p, bg=BG_PANEL); vis_frm.pack(padx=20, pady=6)
        frm_orig, self._lbl_orig, self._info_orig = image_card(vis_frm, "Original")
        frm_orig.grid(row=0, column=0, padx=6)
        frm_res, self._lbl_res, self._info_res = image_card(vis_frm, "Resultado")
        frm_res.grid(row=0, column=1, padx=6)

        make_button(p, "💾  Salvar resultado", self._save).pack(pady=8)

    def _on_img_load(self, arr, _path):
        self._img = arr
        show_array_in_label(arr, self._lbl_orig, self._info_orig)

    def _check_img(self):
        if self._img is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return False
        return True

    def _apply(self, op, params):
        if not self._check_img(): return
        try:
            result = C2.composicao_transformacoes(self._img, [{"operacao": op, "params": params}])
            self._result_arr = result
            show_array_in_label(result, self._lbl_res, self._info_res)
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _run_zoom_in(self):
        if not self._check_img(): return
        try:
            fator = float(self._zoom_in_f.get())
            tipo  = self._zoom_in_tipo.get()
            op    = f"zoom_in_{tipo}"
            result = C2.composicao_transformacoes(self._img, [{"operacao": op, "params": {"fator": fator}}])
            self._result_arr = result
            show_array_in_label(result, self._lbl_res, self._info_res)
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _run_zoom_out(self):
        if not self._check_img(): return
        try:
            fator = float(self._zoom_out_f.get())
            tipo  = self._zoom_out_tipo.get()
            op    = f"zoom_out_{tipo}"
            result = C2.composicao_transformacoes(self._img, [{"operacao": op, "params": {"fator": fator}}])
            self._result_arr = result
            show_array_in_label(result, self._lbl_res, self._info_res)
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _add_step(self):
        import json
        op = self._step_op.get()
        raw = self._step_params.get().strip() or "{}"
        try:
            params = json.loads(raw)
        except json.JSONDecodeError:
            messagebox.showerror("Erro", f"Parâmetro inválido (JSON esperado):\n{raw}")
            return
        self._compose_steps.append({"operacao": op, "params": params})
        self._steps_list.insert(tk.END, f"  {len(self._compose_steps):02d}. {op}  {params}")

    def _remove_step(self):
        sel = self._steps_list.curselection()
        if not sel: return
        idx = sel[0]
        self._steps_list.delete(idx)
        self._compose_steps.pop(idx)
        # renumerar exibição
        items = list(self._steps_list.get(0, tk.END))
        self._steps_list.delete(0, tk.END)
        for i, it in enumerate(items):
            self._steps_list.insert(tk.END, f"  {i+1:02d}." + it[5:])

    def _clear_steps(self):
        self._steps_list.delete(0, tk.END)
        self._compose_steps.clear()

    def _run_compose(self):
        if not self._check_img(): return
        if not self._compose_steps:
            messagebox.showinfo("Info", "Adicione pelo menos uma etapa."); return
        try:
            result = C2.composicao_transformacoes(self._img, self._compose_steps)
            self._result_arr = result
            show_array_in_label(result, self._lbl_res, self._info_res)
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _save(self):
        if self._result_arr is None:
            messagebox.showinfo("Info", "Nenhum resultado para salvar."); return
        path = filedialog.asksaveasfilename(defaultextension=".png",
               filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("BMP","*.bmp")])
        if path:
            Image.fromarray(self._result_arr).save(path)
            messagebox.showinfo("Salvo", f"Imagem salva em:\n{path}")


# ─────────────────────────────────────────────
#  PÁGINA C3 — Espaços de Cor
# ─────────────────────────────────────────────

class PageC3(BasePage):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        inner = self.scrollable()
        self._img = None
        self._build(inner)

    def _build(self, p):
        pad = dict(padx=20, pady=4, anchor="w")
        section_title(p, "Conjunto 3 — Espaços de Cor e Pseudocolorização")

        self._sel = ImageSelector(p, "Imagem", on_load=self._on_img_load)
        self._sel.pack(**pad, fill="x")

        # ── 3A Decomposição RGB ──
        section_title(p, "3A — Decomposição Monocromática (RGB)")
        make_button(p, "Decompor em R / G / B", self._run_rgb
                    ).pack(padx=20, pady=4, anchor="w")
        self._rgb_frm = tk.Frame(p, bg=BG_PANEL)
        self._rgb_frm.pack(padx=20, pady=4)

        # ── 3B Espaços de cor ──
        section_title(p, "3B — Espaços de Cor")
        space_frm = tk.Frame(p, bg=BG_PANEL); space_frm.pack(padx=20,pady=4,anchor="w")
        self._space = make_combobox(space_frm, ["CMY", "CMYK", "YUV", "HSV"], width=10)
        self._space.grid(row=0,column=0,padx=(0,10))
        make_button(space_frm, "Mostrar canais", self._run_space).grid(row=0,column=1)
        self._space_frm = tk.Frame(p, bg=BG_PANEL)
        self._space_frm.pack(padx=20, pady=4)

        # ── 3C Fatiamento por Densidade ──
        section_title(p, "3C — Fatiamento por Densidade")
        make_button(p, "Aplicar pseudocor", self._run_density
                    ).pack(padx=20, pady=4, anchor="w")
        self._density_frm = tk.Frame(p, bg=BG_PANEL)
        self._density_frm.pack(padx=20, pady=4)

        # ── 3D Redistribuição de cores ──
        section_title(p, "3D — Redistribuição de Cores (Falsa Cor)")
        make_button(p, "Aplicar falsa cor", self._run_redist
                    ).pack(padx=20, pady=4, anchor="w")
        self._redist_frm = tk.Frame(p, bg=BG_PANEL)
        self._redist_frm.pack(padx=20, pady=4)

    def _on_img_load(self, arr, _path):
        self._img = arr
        # Limpar frames de resultado
        for frm in [self._rgb_frm, self._space_frm,
                    self._density_frm, self._redist_frm]:
            for w in frm.winfo_children():
                w.destroy()

    def _check(self):
        if self._img is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return False
        return True

    def _show_row(self, parent, arrays_titles, max_w=230, max_h=200):
        """Exibe uma linha de image_cards."""
        for widget in parent.winfo_children():
            widget.destroy()
        for i, (arr, title) in enumerate(arrays_titles):
            frm, lbl_img, lbl_info = image_card(parent, title)
            frm.grid(row=0, column=i, padx=5, pady=4, sticky="n")
            show_array_in_label(arr, lbl_img, lbl_info, max_w, max_h)

    def _run_rgb(self):
        if not self._check(): return
        r, g, b, recombined = C3.decouple_rgb(self._img)
        self._show_row(self._rgb_frm,
                       [(self._img,"Original"),(r,"Canal R"),(g,"Canal G"),(b,"Canal B"),(recombined,"Recomposta")])

    def _run_space(self):
        if not self._check(): return
        space = self._space.get()
        try:
            if space == "CMY":
                imgs = C3.get_cmy_tinted(self._img)
                titles = ["Ciano (C)","Magenta (M)","Amarelo (Y)"]
            elif space == "CMYK":
                imgs = C3.get_cmyk_tinted_images(self._img)
                titles = ["Ciano (C)","Magenta (M)","Amarelo (Y)","Preto (K)"]
            elif space == "YUV":
                imgs = C3.get_yuv_tinted_images(self._img)
                titles = ["Luminância (Y)","Crominância U","Crominância V"]
            else:  # HSV
                imgs = C3.get_hsv_clean_images(self._img)
                titles = ["Matiz (H)","Saturação (S)","Valor (V)"]
            self._show_row(self._space_frm, list(zip(imgs, titles)))
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _run_density(self):
        if not self._check(): return
        gray, pseudo = C3.density_slicing(self._img)
        self._show_row(self._density_frm,
                       [(self._img,"Original"),(gray,"Tons de Cinza"),(pseudo,"Pseudocor")])

    def _run_redist(self):
        if not self._check(): return
        falsa = C3.color_redistribution(self._img)
        self._show_row(self._redist_frm,
                       [(self._img,"Original"),(falsa,"Falsa Cor (G→R, B→G, R→B)")])


# ─────────────────────────────────────────────
#  JANELA PRINCIPAL
# ─────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDI Lab — UFERSA / EXA0188")
        self.geometry("1100x720")
        self.minsize(900, 600)
        self.configure(bg=BG_DARK)
        self._build()

    def _build(self):
        # ── Barra de topo ──
        topbar = tk.Frame(self, bg=BG_PANEL, height=48)
        topbar.pack(side="top", fill="x")
        topbar.pack_propagate(False)
        tk.Label(topbar, text="PDI Lab", fg=ACCENT, bg=BG_PANEL,
                 font=("Courier New", 16, "bold")).pack(side="left", padx=18, pady=8)
        tk.Label(topbar, text="Processamento Digital de Imagens  ·  UFERSA",
                 fg=TEXT_DIM, bg=BG_PANEL, font=FONT_MONO).pack(side="left")
        tk.Frame(topbar, bg=BORDER, width=1).pack(side="left", fill="y", padx=12)

        # ── Layout: sidebar + conteúdo ──
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True)

        sidebar = tk.Frame(body, bg=BG_PANEL, width=200)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        self._content = tk.Frame(body, bg=BG_PANEL)
        self._content.pack(side="left", fill="both", expand=True)

        # ── Páginas ──
        self._pages = {
            "C1": PageC1(self._content),
            "C2": PageC2(self._content),
            "C3": PageC3(self._content),
        }
        for page in self._pages.values():
            page.place(relwidth=1, relheight=1)

        # ── Menu lateral ──
        tk.Label(sidebar, text="MÓDULOS", fg=TEXT_DIM, bg=BG_PANEL,
                 font=("Courier New", 8, "bold")).pack(pady=(20,8), padx=16, anchor="w")

        menu_items = [
            ("C1", "Aritmética\n& Lógica",       ACCENT),
            ("C2", "Transformações\nGeométricas", ACCENT2),
            ("C3", "Espaços de Cor\n& Pseudocor", ACCENT3),
        ]

        self._menu_btns = {}
        for key, label, color in menu_items:
            btn = tk.Button(
                sidebar, text=label, command=lambda k=key: self._show(k),
                bg=BG_PANEL, fg=TEXT_SEC, activebackground=BG_CARD,
                activeforeground=color, relief="flat", font=FONT_BTN,
                cursor="hand2", bd=0, padx=16, pady=14, justify="left",
                anchor="w", width=16
            )
            btn.pack(fill="x", pady=2, padx=8)
            self._menu_btns[key] = (btn, color)

        # Separador e versão
        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=12, pady=16)
        tk.Label(sidebar, text="v1.0  |  NumPy + PIL",
                 fg=TEXT_DIM, bg=BG_PANEL, font=("Courier New",7)).pack(padx=16, anchor="w")

        self._show("C1")

    def _show(self, key: str):
        self._active = key
        # Destaca botão ativo
        for k, (btn, color) in self._menu_btns.items():
            if k == key:
                btn.config(bg=BG_CARD, fg=color)
            else:
                btn.config(bg=BG_PANEL, fg=TEXT_SEC)
        self._pages[key].lift()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
