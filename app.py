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
from impl.unid2 import realce, filtering
from impl.unid2.segmentation import watershed


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
ACCENT4   = "#d73a49"
ACCENT5   = "#a31515"
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
    """Converte ndarray para PhotoImage, redimensionando para caber (só reduz, nunca aumenta)."""
    img = Image.fromarray(arr.astype(np.uint8))
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


def array_to_photoimage_exact(arr: np.ndarray) -> ImageTk.PhotoImage:
    """Converte ndarray para PhotoImage SEM redimensionar — tamanho real."""
    return ImageTk.PhotoImage(Image.fromarray(arr.astype(np.uint8)))


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



def show_result_dynamic(arr, lbl_img, lbl_info, max_w=380, max_h=300, title="Resultado"):
    """
    Exibe miniatura inline. Se a imagem for maior que max_w x max_h,
    abre automaticamente uma janela Toplevel scrollavel em tamanho real.
    """
    h, w = arr.shape[:2]
    ch = arr.shape[2] if arr.ndim == 3 else 1
    info_txt = f"{w}\u00d7{h} px | {ch}ch | dtype={arr.dtype}"

    ph_thumb = array_to_photoimage(arr, max_w, max_h)
    lbl_img.config(image=ph_thumb)
    lbl_img.image = ph_thumb

    if w > max_w or h > max_h:
        lbl_info.config(text=info_txt + "  \u2b06 janela aberta", fg=ACCENT2)
        _open_result_window(arr, title)
    else:
        lbl_info.config(text=info_txt, fg=TEXT_DIM)


def array_to_photoimage_exact(arr):
    """Converte ndarray para PhotoImage SEM redimensionar."""
    return ImageTk.PhotoImage(Image.fromarray(arr.astype(np.uint8)))


def _open_result_window(arr, title="Resultado"):
    """Abre Toplevel scrollavel com a imagem em tamanho real. Suporta pan com mouse."""
    win = tk.Toplevel()
    win.title(f"PDI Lab \u2014 {title}")
    win.configure(bg=BG_DARK)

    h, w = arr.shape[:2]
    ch = arr.shape[2] if arr.ndim == 3 else 1

    hdr = tk.Frame(win, bg=BG_PANEL, pady=6)
    hdr.pack(fill="x")
    tk.Label(hdr, text=f"  {title}", fg=ACCENT, bg=BG_PANEL,
             font=FONT_HEADER).pack(side="left")
    tk.Label(hdr, text=f"{w}\u00d7{h} px  |  {ch} canal(is)",
             fg=TEXT_SEC, bg=BG_PANEL, font=FONT_MONO).pack(side="left", padx=16)

    screen_w = win.winfo_screenwidth()
    screen_h = win.winfo_screenheight()
    win_w = min(w + 30, int(screen_w * 0.9))
    win_h = min(h + 100, int(screen_h * 0.9))
    win.geometry(f"{win_w}x{win_h}")

    frm = tk.Frame(win, bg=BG_DARK)
    frm.pack(fill="both", expand=True)

    canvas = tk.Canvas(frm, bg=BG_DARK, bd=0, highlightthickness=0)
    hsb = tk.Scrollbar(frm, orient="horizontal", command=canvas.xview)
    vsb = tk.Scrollbar(frm, orient="vertical",   command=canvas.yview)
    canvas.configure(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")
    canvas.pack(side="left", fill="both", expand=True)

    ph = array_to_photoimage_exact(arr)
    canvas.create_image(0, 0, anchor="nw", image=ph)
    canvas.image = ph
    canvas.configure(scrollregion=(0, 0, w, h))

    canvas.bind("<ButtonPress-1>", lambda e: canvas.scan_mark(e.x, e.y))
    canvas.bind("<B1-Motion>",     lambda e: canvas.scan_dragto(e.x, e.y, gain=1))
    canvas.bind("<MouseWheel>",
        lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))
    canvas.bind("<Shift-MouseWheel>",
        lambda e: canvas.xview_scroll(-1 * (e.delta // 120), "units"))

    btn_frm = tk.Frame(win, bg=BG_PANEL, pady=4)
    btn_frm.pack(fill="x")

    def _save():
        path = filedialog.asksaveasfilename(
            parent=win, defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("BMP","*.bmp")])
        if path:
            Image.fromarray(arr.astype(np.uint8)).save(path)
            messagebox.showinfo("Salvo", f"Imagem salva em:\n{path}", parent=win)

    make_button(btn_frm, "\U0001f4be  Salvar imagem", _save, color=ACCENT2).pack(side="left", padx=12)
    make_button(btn_frm, "\u2715  Fechar", win.destroy, color=ACCENT3).pack(side="left", padx=4)


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
            show_result_dynamic(result, self._lbl_res, self._info_res, title=f"Resultado — {op}")
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
            show_result_dynamic(result, self._lbl_res, self._info_res, title=f"Zoom IN ×{fator}")
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
            show_result_dynamic(result, self._lbl_res, self._info_res, title=f"Zoom OUT ÷{fator}")
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
            show_result_dynamic(result, self._lbl_res, self._info_res, title="Resultado — Composição")
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


class AbaRealce(BasePage):
    """
    Aba dedicada à Unidade 2: Realce de Imagens (C2)
    """
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        inner = self.scrollable()
        self._img = None
        self._build(inner)

    def _build(self, p):
        pad = dict(padx=20, pady=4, anchor="w")
        section_title(p, "Conjunto 2 — Realce de Imagens (Domínio Espacial)")

        self._sel = ImageSelector(p, "Imagem", on_load=self._on_img_load)
        self._sel.pack(**pad, fill="x")

        # ── 2A Transformação Linear (Intervalo) ──
        section_title(p, "2A — Mapeamento de Intervalo Linear")
        lin_ctrl = tk.Frame(p, bg=BG_PANEL)
        lin_ctrl.pack(**pad)
        
        # Variável para controlar o estado do tempo real (Nasce como Falso)
        self._auto_update = tk.BooleanVar(value=False)

        # -- Frame de Entradas (f) --
        frm_f = tk.Frame(lin_ctrl, bg=BG_PANEL)
        frm_f.grid(row=0, column=0, padx=(0, 10))
        
        self._scale_fmin = tk.Scale(frm_f, from_=0, to=254, orient="horizontal", label="f_min", 
                                    bg=BG_PANEL, fg="white", highlightthickness=0, length=120,
                                    command=self._on_slider_change)
        self._scale_fmin.set(0)
        self._scale_fmin.pack(side="left")
        
        self._scale_fmax = tk.Scale(frm_f, from_=1, to=255, orient="horizontal", label="f_max", 
                                    bg=BG_PANEL, fg="white", highlightthickness=0, length=120,
                                    command=self._on_slider_change)
        self._scale_fmax.set(150)
        self._scale_fmax.pack(side="left")

        # -- Frame de Saídas (g) --
        frm_g = tk.Frame(lin_ctrl, bg=BG_PANEL)
        frm_g.grid(row=0, column=1, padx=(0, 10))
        
        self._scale_gmin = tk.Scale(frm_g, from_=0, to=255, orient="horizontal", label="g_min", 
                                    bg=BG_PANEL, fg="white", highlightthickness=0, length=120,
                                    command=self._on_slider_change)
        self._scale_gmin.set(0)
        self._scale_gmin.pack(side="left")
        
        self._scale_gmax = tk.Scale(frm_g, from_=0, to=255, orient="horizontal", label="g_max", 
                                    bg=BG_PANEL, fg="white", highlightthickness=0, length=120,
                                    command=self._on_slider_change)
        self._scale_gmax.set(255)
        self._scale_gmax.pack(side="left")

        # -- Frame de Controles (Aplicar, Resetar, Auto-Update) --
        frm_btn = tk.Frame(lin_ctrl, bg=BG_PANEL)
        frm_btn.grid(row=0, column=2, sticky="s", pady=15)
        
        make_button(frm_btn, "Aplicar", self._run_intervalo).pack(fill="x", pady=2)
        make_button(frm_btn, "Resetar", self._resetar_intervalo).pack(fill="x", pady=2)
        
        # O checkbox que controla o auto-update
        chk_auto = tk.Checkbutton(frm_btn, text="Tempo real", variable=self._auto_update, 
                                  bg=BG_PANEL, fg="white", selectcolor=BG_DARK, 
                                  activebackground=BG_PANEL, activeforeground="white")
        chk_auto.pack(anchor="w")

        self._lin_res_frm = tk.Frame(p, bg=BG_PANEL)
        self._lin_res_frm.pack(padx=20, pady=4)

        # ── 2B Inversa (Negativo) ──
        section_title(p, "2B — Transformação Inversa (Negativo)")
        inv_ctrl = tk.Frame(p, bg=BG_PANEL)
        inv_ctrl.pack(**pad)
        make_button(inv_ctrl, "Gerar Negativo", self._run_inversa).pack(side="left")
        
        self._inv_res_frm = tk.Frame(p, bg=BG_PANEL)
        self._inv_res_frm.pack(padx=20, pady=4)

        # ── 2C Binária (Limiarização) ──
        section_title(p, "2C — Transformação Binária (Threshold)")
        bin_ctrl = tk.Frame(p, bg=BG_PANEL)
        bin_ctrl.pack(**pad)
        
        tk.Label(bin_ctrl, text="Limiar (T):", bg=BG_PANEL, fg="white").pack(side="left")
        
        # Variável para controlar o tempo real da Binária
        self._auto_bin = tk.BooleanVar(value=False)
        
        # Slider do Limiar
        self._scale_limiar = tk.Scale(bin_ctrl, from_=0, to=255, orient="horizontal", 
                                      bg=BG_PANEL, fg="white", highlightthickness=0, length=200,
                                      command=self._on_bin_slider_change)
        self._scale_limiar.set(127)
        self._scale_limiar.pack(side="left", padx=10)
        
        # Botões alinhados na horizontal
        make_button(bin_ctrl, "Aplicar", self._run_binaria).pack(side="left", padx=5)
        
        chk_auto_bin = tk.Checkbutton(bin_ctrl, text="Tempo real", variable=self._auto_bin, 
                                      bg=BG_PANEL, fg="white", selectcolor=BG_DARK, 
                                      activebackground=BG_PANEL, activeforeground="white")
        chk_auto_bin.pack(side="left", padx=5)
        
        self._bin_res_frm = tk.Frame(p, bg=BG_PANEL)
        self._bin_res_frm.pack(padx=20, pady=4)

        # ── 2D Transformações Não Lineares ──
        # ── 2D Transformações Não Lineares ──
        section_title(p, "2D — Transformações Não Lineares")
        nl_ctrl = tk.Frame(p, bg=BG_PANEL)
        nl_ctrl.pack(**pad)
        
        make_button(nl_ctrl, "Exibir todas", self._run_nao_linear).pack(side="left")
        
        self._nl_res_frm = tk.Frame(p, bg=BG_PANEL)
        self._nl_res_frm.pack(padx=20, pady=4)

        # ── 2E Correção Gama ──
        section_title(p, "2E — Correção Gama")
        gam_ctrl = tk.Frame(p, bg=BG_PANEL)
        gam_ctrl.pack(**pad)
        
        tk.Label(gam_ctrl, text="Gama (γ):", bg=BG_PANEL, fg="white").pack(side="left")
        
        # Variável para o tempo real da Correção Gama
        self._auto_gama = tk.BooleanVar(value=False)
        
        # Slider para Gama (Permite valores de 0.1 a 5.0 com resolução de 0.1)
        self._scale_gama = tk.Scale(gam_ctrl, from_=0.1, to=5.0, resolution=0.1, orient="horizontal", 
                                    bg=BG_PANEL, fg="white", highlightthickness=0, length=200,
                                    command=self._on_gama_slider_change)
        self._scale_gama.set(1.0) # Gama 1.0 = Imagem original
        self._scale_gama.pack(side="left", padx=10)
        
        make_button(gam_ctrl, "Aplicar", self._run_gama).pack(side="left", padx=5)
        
        chk_auto_gama = tk.Checkbutton(gam_ctrl, text="Tempo real", variable=self._auto_gama, 
                                       bg=BG_PANEL, fg="white", selectcolor=BG_DARK, 
                                       activebackground=BG_PANEL, activeforeground="white")
        chk_auto_gama.pack(side="left", padx=5)
        
        self._gam_res_frm = tk.Frame(p, bg=BG_PANEL)
        self._gam_res_frm.pack(padx=20, pady=4)

        # ── 2F Equalização de Histograma ──
        section_title(p, "2F — Equalização de Histograma")
        eq_ctrl = tk.Frame(p, bg=BG_PANEL)
        eq_ctrl.pack(**pad)
        
        make_button(eq_ctrl, "Equalizar Imagem Automático", self._run_equalizacao).pack(side="left")
        
        # Frame de resultados exclusivo para a equalização
        self._eq_res_frm = tk.Frame(p, bg=BG_PANEL)
        self._eq_res_frm.pack(padx=20, pady=4)

        # ── 2G Fatiamento de Bits ──
        section_title(p, "2G — Fatiamento de Planos de Bits")
        bit_ctrl = tk.Frame(p, bg=BG_PANEL)
        bit_ctrl.pack(**pad)
        make_button(bit_ctrl, "Extrair 8 Planos de Bits", self._run_fatiamento).pack(side="left")
        
        self._bit_res_frm = tk.Frame(p, bg=BG_PANEL)
        self._bit_res_frm.pack(padx=20, pady=4)

    # =========================================================================
    # Helpers e Validações
    # =========================================================================

    def _on_img_load(self, arr, _path):
        self._img = arr
        # Adicionamos o self._eq_res_frm na lista de limpeza!
        frms = [self._lin_res_frm, self._inv_res_frm, self._bin_res_frm, 
                self._nl_res_frm, self._gam_res_frm, self._eq_res_frm, self._bit_res_frm]
        for frm in frms:
            for w in frm.winfo_children(): w.destroy()

    def _get_gray_img(self):
        if self._img is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return None
        if len(self._img.shape) == 3:
            return np.array(Image.fromarray(self._img).convert("L"))
        return self._img

    def _show_grid(self, parent, arrays_titles, cols=4, max_w=230, max_h=200):
        for widget in parent.winfo_children(): widget.destroy()
        for i, (arr, title) in enumerate(arrays_titles):
            frm, lbl_img, lbl_info = image_card(parent, title)
            row, col = i // cols, i % cols
            frm.grid(row=row, column=col, padx=5, pady=4, sticky="n")
            show_array_in_label(arr, lbl_img, lbl_info, max_w, max_h)

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _run_intervalo(self, *args):
        arr = self._get_gray_img()
        if arr is None: return
        
        # Lê os valores direto dos sliders (já vêm como inteiros)
        fmin = self._scale_fmin.get()
        fmax = self._scale_fmax.get()
        gmin = self._scale_gmin.get()
        gmax = self._scale_gmax.get()
        
        # Trava de segurança: impede que f_min e f_max sejam idênticos (divisão por zero)
        if fmin == fmax:
            if fmax < 255: 
                fmax += 1
                self._scale_fmax.set(fmax)
            else: 
                fmin -= 1
                self._scale_fmin.set(fmin)

        try:
            res = realce.transformacao_linear_intervalo(arr, fmin, fmax, gmin, gmax)
            # Mostra o intervalo escolhido no título dinamicamente
            titulo = f"Intervalo: [{fmin}, {fmax}] → [{gmin}, {gmax}]"
            self._show_grid(self._lin_res_frm, [(arr, "Original"), (res, titulo)])
        except Exception as e: 
            messagebox.showerror("Erro", str(e))
    
    def _on_slider_change(self, *args):
        """Callback disparado toda vez que um slider move um milímetro."""
        # Só executa automaticamente se a caixinha estiver marcada E a imagem existir
        if self._auto_update.get() and self._img is not None:
            self._run_intervalo()

    def _resetar_intervalo(self):
        """Volta os sliders para o padrão sem estourar o layout"""
        self._scale_fmin.set(0)
        self._scale_fmax.set(255)
        self._scale_gmin.set(0)
        self._scale_gmax.set(255)
        # O set() do slider já vai chamar o _run_intervalo automaticamente!

    def _run_inversa(self):
        arr = self._get_gray_img()
        if arr is None: return
        res = realce.transformacao_inversa(arr)
        self._show_grid(self._inv_res_frm, [(arr, "Original"), (res, "Negativo")])
    
    def _on_bin_slider_change(self, *args):
        """Callback disparado toda vez que o slider do limiar se move."""
        # Garante que as variáveis já foram inicializadas e a imagem existe
        if hasattr(self, '_auto_bin') and self._auto_bin.get() and self._img is not None:
            self._run_binaria()

    def _run_binaria(self):
        arr = self._get_gray_img()
        if arr is None: return
        try:
            # Agora lemos o valor de corte diretamente do slider
            t = self._scale_limiar.get()
            res = realce.transformacao_binaria(arr, t)
            self._show_grid(self._bin_res_frm, [(arr, "Original"), (res, f"Binária T={t}")])
        except Exception as e: 
            messagebox.showerror("Erro", str(e))

    def _run_nao_linear(self, *args):
        arr = self._get_gray_img()
        if arr is None: return
        
        try:
            # Calcula todas as 4 transformações de uma vez
            res_log = realce.transformacao_logaritmica(arr)
            res_raiz = realce.transformacao_raiz(arr)
            res_exp = realce.transformacao_exponencial(arr)
            res_quad = realce.transformacao_quadrado(arr)
            
            # Monta a lista com os títulos para a grade
            comparativos = [
                (arr, "Original"),
                (res_log, "Logarítmica"),
                (res_raiz, "Raiz Quadrada"),
                (res_exp, "Exponencial"),
                (res_quad, "Quadrado")
            ]
            
            # O cols=3 vai colocar 3 imagens na primeira linha e 2 na segunda linha.
            # Reduzi um pouco o max_w e max_h para garantir que caibam bem na tela.
            self._show_grid(self._nl_res_frm, comparativos, cols=3, max_w=180, max_h=180)
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _on_gama_slider_change(self, *args):
        """Callback disparado toda vez que o slider do Gama se move."""
        if hasattr(self, '_auto_gama') and self._auto_gama.get() and self._img is not None:
            self._run_gama()

    def _run_gama(self, *args):
        arr = self._get_gray_img()
        if arr is None: return
        try:
            # Pega o valor do slider (float)
            g = float(self._scale_gama.get())
            res = realce.correcao_gama(arr, g)
            self._show_grid(self._gam_res_frm, [(arr, "Original"), (res, f"Gama (y={g})")])
        except Exception as e: 
            messagebox.showerror("Erro", str(e))

    def _run_equalizacao(self):
        arr = self._get_gray_img()
        if arr is None: return
        
        try:
            # 1. Processa a imagem
            res = realce.equalizar_histograma(arr)
            
            # 2. Gera os gráficos dos histogramas
            hist_orig = self._plot_hist_to_array(arr)
            hist_eq = self._plot_hist_to_array(res)
            
            # 3. Monta a lista comparativa
            # Organizado para que na grade 2x2 fiquem:
            # Imagem Original | Imagem Equalizada
            # Hist. Original  | Hist. Equalizado
            comparativo = [
                (arr, "Imagem Original"),
                (res, "Imagem Equalizada"),
                (hist_orig, "Histograma Original"),
                (hist_eq, "Histograma Equalizado")
            ]
            
            # Exibe em grade de 2 colunas
            self._show_grid(self._eq_res_frm, comparativo, cols=2, max_w=350, max_h=250)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha na equalização: {e}")

    def _run_fatiamento(self):
        arr = self._get_gray_img()
        if arr is None: return
        planos = realce.fatiamento_planos_bits(arr)
        planos.reverse()
        titles = [f"Bit {i}" for i in range(7, -1, -1)]
        self._show_grid(self._bit_res_frm, list(zip(planos, titles)), cols=4, max_w=150, max_h=150)
    
    def _plot_hist_to_array(self, arr):
        """Gera um array de imagem contendo o gráfico do histograma."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        # Criamos uma figura pequena e com o fundo do seu painel
        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        fig.patch.set_facecolor('#161b22') # BG_PANEL
        ax.set_facecolor('#161b22')

        # Desenha o histograma
        ax.hist(arr.flatten(), bins=256, range=[0, 256], color='#58a6ff', alpha=0.8)
        
        # Estética do gráfico (cores dos eixos para combinar com o tema Dark)
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d') # BORDER

        ax.set_xlim([0, 256])
        plt.tight_layout()

        # Converte o desenho do Matplotlib para um array do NumPy (RGB)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba_buffer = canvas.buffer_rgba()
        res_array = np.array(rgba_buffer)
        
        plt.close(fig) # Fecha a figura para não consumir memória
        return res_array

class AbaFiltragem(BasePage):
    """
    Aba dedicada à Filtragem Espacial e Meios-Tons.
    """
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        inner = self.scrollable()
        self._img = None
        self._build(inner)

    def _build(self, p):
        pad = dict(padx=20, pady=4, anchor="w")
        section_title(p, "Módulo de Filtragem Espacial e Halftoning")

        self._sel = ImageSelector(p, "Imagem", on_load=self._on_img_load)
        self._sel.pack(**pad, fill="x")

        # ── 1. Filtros Passa-Baixa (Suavização e Morfologia) ──
        section_title(p, "1 — Filtros Passa-Baixa (Suavização e Morfologia)")
        pb_ctrl = tk.Frame(p, bg=BG_PANEL)
        pb_ctrl.pack(**pad)
        
        self._cb_pb = make_combobox(pb_ctrl, ["Média", "Mediana", "Máximo", "Mínimo", "Moda"], width=15)
        self._cb_pb.pack(side="left", padx=(0, 10))
        
        tk.Label(pb_ctrl, text="Kernel:", bg=BG_PANEL, fg="white").pack(side="left")
        self._cb_kernel = make_combobox(pb_ctrl, ["3", "5", "7", "9"], width=5)
        self._cb_kernel.current(0) # Padrão 3x3
        self._cb_kernel.pack(side="left", padx=5)
        
        make_button(pb_ctrl, "Aplicar", self._run_passa_baixa).pack(side="left", padx=10)
        
        self._pb_res_frm = tk.Frame(p, bg=BG_PANEL)
        self._pb_res_frm.pack(padx=20, pady=4)

        # ── 2. Filtros Passa-Baixa (Preservação de Bordas) ──
        section_title(p, "2 — Filtros de Preservação de Bordas (Janela 5x5)")
        pres_ctrl = tk.Frame(p, bg=BG_PANEL)
        pres_ctrl.pack(**pad)
        
        self._cb_pres = make_combobox(pres_ctrl, ["Kuwahara", "Tomita e Tsuji", "Nagao e Matsuyama", "Somboonkaew"], width=20)
        self._cb_pres.pack(side="left", padx=(0, 10))
        
        make_button(pres_ctrl, "Aplicar", self._run_preservacao).pack(side="left")
        
        self._pres_res_frm = tk.Frame(p, bg=BG_PANEL)
        self._pres_res_frm.pack(padx=20, pady=4)

        # ── 3. Filtros Passa-Alta ──
        section_title(p, "3 — Filtros Passa-Alta (Detenção de Bordas)")
        pa_ctrl = tk.Frame(p, bg=BG_PANEL)
        pa_ctrl.pack(**pad)
        
        self._cb_pa = make_combobox(pa_ctrl, ["H1", "H2", "M1", "M2", "M3"], width=10)
        self._cb_pa.pack(side="left", padx=(0, 10))
        
        make_button(pa_ctrl, "Aplicar Máscara", self._run_passa_alta).pack(side="left")
        
        self._pa_res_frm = tk.Frame(p, bg=BG_PANEL)
        self._pa_res_frm.pack(padx=20, pady=4)

        # ── 4 — Alto-Reforço (High-Boost) ──
        section_title(p, "4 — Alto-Reforço (High-Boost)")
        hb_ctrl = tk.Frame(p, bg=BG_PANEL)
        hb_ctrl.pack(**pad)
        
        # Variável para controlar o tempo real
        self._auto_hb = tk.BooleanVar(value=False)

        tk.Label(hb_ctrl, text="Fator A:", bg=BG_PANEL, fg="white").pack(side="left")
        
        # Slider atualizado com o comando de mudança
        self._scale_a = tk.Scale(hb_ctrl, from_=1.0, to=3.0, resolution=0.1, orient="horizontal", 
                                 bg=BG_PANEL, fg="white", highlightthickness=0, length=200,
                                 command=self._on_hb_slider_change)
        self._scale_a.set(1.2)
        self._scale_a.pack(side="left", padx=10)
        
        # Botões e Checkbox alinhados na horizontal
        make_button(hb_ctrl, "Aplicar", self._run_alto_reforco).pack(side="left", padx=5)
        make_button(hb_ctrl, "Resetar", self._resetar_alto_reforco).pack(side="left", padx=5)
        
        chk_auto_hb = tk.Checkbutton(hb_ctrl, text="Tempo real", variable=self._auto_hb, 
                                     bg=BG_PANEL, fg="white", selectcolor=BG_DARK, 
                                     activebackground=BG_PANEL, activeforeground="white")
        chk_auto_hb.pack(side="left", padx=5)
        
        self._hb_res_frm = tk.Frame(p, bg=BG_PANEL)
        self._hb_res_frm.pack(padx=20, pady=4)

        # ── 5. Meios-Tons: Pontilhado Ordenado ──
        section_title(p, "5 — Meios-Tons: Pontilhado Ordenado")
        po_ctrl = tk.Frame(p, bg=BG_PANEL)
        po_ctrl.pack(**pad)
        
        self._cb_po = make_combobox(po_ctrl, ["2x2", "2x3", "3x3"], width=10)
        self._cb_po.pack(side="left", padx=(0, 10))
        make_button(po_ctrl, "Gerar Pontilhado", self._run_pontilhado_ordenado).pack(side="left")
        
        self._po_res_frm = tk.Frame(p, bg=BG_PANEL)
        self._po_res_frm.pack(padx=20, pady=4)

        # ── 6. Meios-Tons: Difusão de Erro ──
        section_title(p, "6 — Meios-Tons: Difusão de Erro")
        dif_ctrl = tk.Frame(p, bg=BG_PANEL)
        dif_ctrl.pack(**pad)
        
        self._cb_dif = make_combobox(dif_ctrl, [
            "Floyd e Steinberg", "Rogers", "Jarvis, Judice & Ninke", "Stucki", "Stevenson e Arce"
        ], width=25)
        self._cb_dif.pack(side="left", padx=(0, 10))
        make_button(dif_ctrl, "Processar Difusão", self._run_difusao).pack(side="left")
        
        tk.Label(dif_ctrl, text="(Atenção: Processo sequencial lento)", bg=BG_PANEL, fg=TEXT_SEC, font=("Courier New", 8)).pack(side="left", padx=10)
        
        self._dif_res_frm = tk.Frame(p, bg=BG_PANEL)
        self._dif_res_frm.pack(padx=20, pady=4)

    # =========================================================================
    # Utilitários Visuais
    # =========================================================================

    def _on_img_load(self, arr, _path):
        self._img = arr
        # Limpar todos os frames de resultado ao carregar uma nova imagem
        frms = [self._pb_res_frm, self._pres_res_frm, self._pa_res_frm, 
                self._hb_res_frm, self._po_res_frm, self._dif_res_frm]
        for frm in frms:
            for w in frm.winfo_children(): w.destroy()

    def _get_gray_img(self):
        """Converte de forma segura para tons de cinzento."""
        if self._img is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return None
        if len(self._img.shape) == 3:
            return np.array(Image.fromarray(self._img).convert("L"))
        return self._img

    def _show_grid(self, parent, arrays_titles, cols=4, max_w=230, max_h=200):
        for widget in parent.winfo_children(): widget.destroy()
        for i, (arr, title) in enumerate(arrays_titles):
            frm, lbl_img, lbl_info = image_card(parent, title)
            row, col = i // cols, i % cols
            frm.grid(row=row, column=col, padx=5, pady=4, sticky="n")
            show_array_in_label(arr, lbl_img, lbl_info, max_w, max_h)

    # =========================================================================
    # Callbacks (Ligação com o filtering.py)
    # =========================================================================

    def _run_passa_baixa(self):
        arr = self._get_gray_img()
        if arr is None: return
        
        tipo = self._cb_pb.get()
        k = int(self._cb_kernel.get())
        
        mapa_filtros = {
            "Média": filtering.filtro_media,
            "Mediana": filtering.filtro_mediana,
            "Máximo": filtering.filtro_maximo,
            "Mínimo": filtering.filtro_minimo,
            "Moda": filtering.filtro_moda
        }
        
        try:
            res = mapa_filtros[tipo](arr, kernel_size=k)
            self._show_grid(self._pb_res_frm, [(arr, "Original"), (res, f"{tipo} ({k}x{k})")])
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _run_preservacao(self):
        arr = self._get_gray_img()
        if arr is None: return
        
        tipo = self._cb_pres.get()
        mapa_pres = {
            "Kuwahara": filtering.filtro_kuwahara,
            "Tomita e Tsuji": filtering.filtro_tomita_tsuji,
            "Nagao e Matsuyama": filtering.filtro_nagao_matsuyama,
            "Somboonkaew": filtering.filtro_somboonkaew
        }
        
        try:
            res = mapa_pres[tipo](arr)
            self._show_grid(self._pres_res_frm, [(arr, "Original"), (res, f"{tipo}")])
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _run_passa_alta(self):
        arr = self._get_gray_img()
        if arr is None: return
        
        tipo = self._cb_pa.get()
        try:
            res = filtering.aplicar_filtro_passa_alta(arr, tipo)
            self._show_grid(self._pa_res_frm, [(arr, "Original"), (res, f"Máscara {tipo}")])
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _on_hb_slider_change(self, *args):
        """Callback para atualização automática do High-Boost."""
        if hasattr(self, '_auto_hb') and self._auto_hb.get() and self._img is not None:
            self._run_alto_reforco()

    def _resetar_alto_reforco(self):
        """Reseta o fator A para o padrão de 1.2."""
        self._scale_a.set(1.2)
        # Se o tempo real estiver ativo, o .set() já dispara a atualização

    def _run_alto_reforco(self, *args):
        """Executa o filtro de Alto-Reforço."""
        arr = self._get_gray_img()
        if arr is None: return
        
        a_val = self._scale_a.get()
        try:
            res = filtering.filtro_alto_reforco(arr, A=a_val)
            self._show_grid(self._hb_res_frm, [(arr, "Original"), (res, f"High-Boost (A={a_val})")])
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _run_pontilhado_ordenado(self):
        arr = self._get_gray_img()
        if arr is None: return
        
        tipo = self._cb_po.get()
        mapa_po = {
            "2x2": filtering.pontilhado_ordenado_2x2,
            "2x3": filtering.pontilhado_ordenado_2x3,
            "3x3": filtering.pontilhado_ordenado_3x3
        }
        
        try:
            res = mapa_po[tipo](arr)
            self._show_grid(self._po_res_frm, [(arr, "Original"), (res, f"Ordenado {tipo}")])
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _run_difusao(self):
        arr = self._get_gray_img()
        if arr is None: return
        
        tipo = self._cb_dif.get()
        mapa_dif = {
            "Floyd e Steinberg": filtering.difusao_floyd_steinberg,
            "Rogers": filtering.difusao_rogers,
            "Jarvis, Judice & Ninke": filtering.difusao_jarvis_judice_ninke,
            "Stucki": filtering.difusao_stucki,
            "Stevenson e Arce": filtering.difusao_stevenson_arce
        }
        
        # Como o processo demora, forçamos o update da interface para evitar que o botão pareça "congelado"
        self.app.update_idletasks()
        
        try:
            res = mapa_dif[tipo](arr)
            self._show_grid(self._dif_res_frm, [(arr, "Original"), (res, f"Difusão: {tipo}")])
        except Exception as e:
            messagebox.showerror("Erro", str(e))


# ─────────────────────────────────────────────
#  PÁGINA DE SEGMENTAÇÃO
# ─────────────────────────────────────────────

class AbaSegmentacao(BasePage):
    """Página de Segmentação — usa impl/unid2/segmentation.py"""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        from impl.unid2 import segmentation as seg
        self._seg = seg
        self._img = None
        self._last_result = None
        inner = self.scrollable()
        self._build(inner)

    # ── Layout ───────────────────────────────
    def _build(self, p):
        pad = dict(padx=20, pady=4, anchor="w")
        section_title(p, "Segmentação de Imagens")

        self._sel = ImageSelector(p, "Imagem", on_load=self._on_img)
        self._sel.pack(**pad, fill="x")

        self._build_pontos(p, pad)
        self._build_retas(p, pad)
        self._build_bordas(p, pad)
        self._build_limiar(p, pad)
        self._build_regioes(p, pad)

        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=20, pady=10)
        make_button(p, "💾  Salvar último resultado",
                    self._save, color=TEXT_SEC).pack(padx=20, anchor="w", pady=(0, 16))

    def _on_img(self, arr, _path=None):
        self._img = arr
        # Atualiza o canvas clicável de sementes sempre que uma nova imagem é carregada
        if hasattr(self, "_canvas_reg"):
            self._sementes_pixels.clear()
            self._render_canvas_img()
            self._atualizar_lbl_sementes()

    def _check(self):
        if self._img is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return False
        return True

    # ── Helpers visuais ───────────────────────
    def _show_row(self, parent, pairs, max_w=300, max_h=240):
        """Limpa parent e exibe uma linha de image_cards com os arrays."""
        for w in parent.winfo_children():
            w.destroy()
        for col, (arr, title) in enumerate(pairs):
            frm, lbl, info = image_card(parent, title)
            frm.grid(row=0, column=col, padx=5, pady=4, sticky="n")
            show_array_in_label(arr, lbl, info, max_w, max_h)

    def _two_cards(self, parent, orig, result, title_res):
        """Exibe par original + resultado."""
        self._show_row(parent, [(orig, "Original"), (result, title_res)])
        self._last_result = result

    # ─────────────────────────────────────────
    #  1 — Detecção de Pontos
    # ─────────────────────────────────────────
    def _build_pontos(self, p, pad):
        section_title(p, "1 — Detecção de Pontos")

        ctrl = tk.Frame(p, bg=BG_PANEL); ctrl.pack(**pad)
        self._scale_ponto = tk.Scale(
            ctrl, from_=0, to=255, orient="horizontal", label="T (limiar)",
            bg=BG_PANEL, fg="white", highlightthickness=0, length=260,
            command=self._on_ponto_slider
        )
        self._scale_ponto.set(50)
        self._scale_ponto.pack(side="left")

        self._frm_pontos = tk.Frame(p, bg=BG_PANEL); self._frm_pontos.pack(padx=20, pady=4)

    def _on_ponto_slider(self, _value=None):
        """Atualiza detecção de pontos em tempo real conforme o slider move."""
        if self._img is None:
            return
        try:
            T = self._scale_ponto.get()
            res = self._seg.points_detection(self._img, T)
            self._two_cards(self._frm_pontos, self._img, res,
                            f"Pontos detectados  (T={T})")
        except Exception:
            pass

    def _run_pontos(self):
        self._on_ponto_slider()

    # ─────────────────────────────────────────
    #  2 — Detecção de Retas
    # ─────────────────────────────────────────
    def _build_retas(self, p, pad):
        section_title(p, "2 — Detecção de Retas")
        frm = tk.Frame(p, bg=BG_PANEL); frm.pack(**pad)
        self._reta_dir = make_combobox(frm,
            ["horizontal", "vertical", "45graus", "135graus"], width=14)
        self._reta_dir.grid(row=0, column=0, padx=(0, 10))
        make_button(frm, "Detectar", self._run_retas).grid(row=0, column=1)

        self._frm_retas = tk.Frame(p, bg=BG_PANEL); self._frm_retas.pack(padx=20, pady=4)

    def _run_retas(self):
        if not self._check(): return
        try:
            res = self._seg.lines_detection(self._img, self._reta_dir.get())
            self._two_cards(self._frm_retas, self._img, res,
                            f"Retas — {self._reta_dir.get()}")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    # ─────────────────────────────────────────
    #  3 — Detecção de Bordas
    # ─────────────────────────────────────────
    def _build_bordas(self, p, pad):
        section_title(p, "3 — Detecção de Bordas")
        frm = tk.Frame(p, bg=BG_PANEL); frm.pack(**pad)
        self._borda_met = make_combobox(frm, [
            "roberts", "roberts_cruzado",
            "prewitt", "sobel",
            "kirsch", "robinson",
            "frei_chen",
            "laplaciano_h1", "laplaciano_h2"
        ], width=18)
        self._borda_met.grid(row=0, column=0, padx=(0, 10))
        make_button(frm, "Calcular", self._run_bordas).grid(row=0, column=1)

        self._frm_bordas = tk.Frame(p, bg=BG_PANEL)
        self._frm_bordas.pack(padx=20, pady=4)

    def _run_bordas(self):
        if not self._check(): return
        try:
            met = self._borda_met.get()
            resultado = self._seg.edges_detection(self._img, met)
            # Monta pares: original + cada saída (gx, gy, magnitude)
            pares = [("Original", self._img)] + [
                (k.upper(), v) for k, v in resultado.items()
            ]
            self._show_row(self._frm_bordas, [(arr, t) for t, arr in pares])
            # Salva magnitude como último resultado
            self._last_result = resultado.get("magnitude",
                                 list(resultado.values())[-1])
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    # ─────────────────────────────────────────
    #  4 — Limiarização
    # ─────────────────────────────────────────
    def _build_limiar(self, p, pad):
        section_title(p, "4 — Limiarização")

        # Global
        frm_g = tk.Frame(p, bg=BG_PANEL); frm_g.pack(**pad)
        tk.Label(frm_g, text="Global (iterativa):", fg=TEXT_SEC,
                 bg=BG_PANEL, font=FONT_LABEL).grid(row=0, column=0, padx=(0, 8))
        make_button(frm_g, "Aplicar", self._run_global).grid(row=0, column=1)
        self._lbl_T_global = tk.Label(frm_g, text="", fg=ACCENT2,
                                       bg=BG_PANEL, font=FONT_MONO)
        self._lbl_T_global.grid(row=0, column=2, padx=8)

        # Local
        frm_l = tk.Frame(p, bg=BG_PANEL); frm_l.pack(**pad)
        tk.Label(frm_l, text="Local — modo:", fg=TEXT_SEC,
                 bg=BG_PANEL, font=FONT_LABEL).grid(row=0, column=0)
        self._local_modo = make_combobox(frm_l,
            ["media", "maximo", "minimo", "niblack"], width=10)
        self._local_modo.grid(row=0, column=1, padx=6)
        tk.Label(frm_l, text="n:", fg=TEXT_SEC, bg=BG_PANEL,
                 font=FONT_LABEL).grid(row=0, column=2)
        self._local_n = make_entry(frm_l, width=4); self._local_n.insert(0, "15")
        self._local_n.grid(row=0, column=3, padx=4)
        tk.Label(frm_l, text="k:", fg=TEXT_SEC, bg=BG_PANEL,
                 font=FONT_LABEL).grid(row=0, column=4)
        self._local_k = make_entry(frm_l, width=5); self._local_k.insert(0, "-0.2")
        self._local_k.grid(row=0, column=5, padx=4)
        make_button(frm_l, "Aplicar", self._run_local).grid(row=0, column=6, padx=8)

        self._frm_limiar = tk.Frame(p, bg=BG_PANEL)
        self._frm_limiar.pack(padx=20, pady=4)

    def _run_global(self):
        if not self._check(): return
        try:
            res, T = self._seg.global_limiarization(self._img)
            self._lbl_T_global.config(text=f"T = {T}")
            self._two_cards(self._frm_limiar, self._img, res, "Global limiarizada")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _run_local(self):
        if not self._check(): return
        try:
            n    = int(self._local_n.get())
            k    = float(self._local_k.get())
            modo = self._local_modo.get()
            res  = self._seg.local_limiarization(self._img, n, modo, k)
            self._two_cards(self._frm_limiar, self._img, res,
                            f"Local — {modo}  n={n}  k={k}")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    # ─────────────────────────────────────────
    #  5 — Segmentação por Regiões
    # ─────────────────────────────────────────
    def _build_regioes(self, p, pad):
        section_title(p, "5 — Segmentação por Regiões")

        # ── Crescimento de Região ──
        tk.Label(p, text="Crescimento de Região", fg=ACCENT2,
                 bg=BG_PANEL, font=FONT_LABEL).pack(**pad)

        # Controles: T + botões
        frm_ctrl = tk.Frame(p, bg=BG_PANEL); frm_ctrl.pack(**pad)

        tk.Label(frm_ctrl, text="T (limiar):", fg=TEXT_SEC,
                 bg=BG_PANEL, font=FONT_LABEL).grid(row=0, column=0, padx=(0, 4))
        self._scale_reg = tk.Scale(
            frm_ctrl, from_=0, to=100, orient="horizontal",
            bg=BG_PANEL, fg="white", highlightthickness=0, length=200,
        )
        self._scale_reg.set(20)
        self._scale_reg.grid(row=0, column=1, padx=(0, 12))

        make_button(frm_ctrl, "Limpar sementes", self._limpar_sementes,
                    color=ACCENT3).grid(row=0, column=2, padx=4)
        make_button(frm_ctrl, "▶ Calcular regiões", self._run_regioes,
                    color=ACCENT2).grid(row=0, column=3, padx=4)

        # Dica de uso
        tk.Label(p, text="  Clique na imagem para adicionar sementes. Clique com botão direito para remover a última.",
                 fg=TEXT_DIM, bg=BG_PANEL, font=FONT_MONO).pack(anchor="w", padx=20, pady=(0, 4))

        # Frame com canvas clicavel (esquerda) + resultado (direita)
        frm_painel = tk.Frame(p, bg=BG_PANEL); frm_painel.pack(padx=20, pady=4, anchor="w")

        # -- Lado esquerdo: canvas da imagem --
        frm_canvas = tk.Frame(frm_painel, bg=BG_CARD,
                               highlightthickness=1, highlightbackground=BORDER)
        frm_canvas.grid(row=0, column=0, padx=(0, 10), sticky="n")

        tk.Label(frm_canvas, text="Clique para adicionar sementes",
                 fg=TEXT_SEC, bg=BG_CARD, font=FONT_MONO).pack(pady=(6, 2))

        self._CANVAS_W = 320
        self._CANVAS_H = 260
        self._canvas_reg = tk.Canvas(
            frm_canvas, width=self._CANVAS_W, height=self._CANVAS_H,
            bg=BG_DARK, bd=0, highlightthickness=0, cursor="crosshair"
        )
        self._canvas_reg.pack(padx=6, pady=4)

        self._lbl_sementes_info = tk.Label(
            frm_canvas, text="Sementes: nenhuma", fg=ACCENT2,
            bg=BG_CARD, font=FONT_MONO
        )
        self._lbl_sementes_info.pack(pady=(0, 6))

        # -- Lado direito: resultado --
        frm_res = tk.Frame(frm_painel, bg=BG_CARD,
                            highlightthickness=1, highlightbackground=BORDER)
        frm_res.grid(row=0, column=1, sticky="n")

        tk.Label(frm_res, text="Resultado", fg=TEXT_SEC,
                 bg=BG_CARD, font=FONT_MONO).pack(pady=(6, 2))
        self._lbl_reg_res = tk.Label(frm_res, bg=BG_CARD,
                                      width=self._CANVAS_W, height=self._CANVAS_H)
        self._lbl_reg_res.pack(padx=6, pady=4)
        self._lbl_reg_info = tk.Label(frm_res, text="—", fg=TEXT_DIM,
                                       bg=BG_CARD, font=FONT_MONO)
        self._lbl_reg_info.pack(pady=(0, 6))

        # Estado interno do canvas clicavel
        self._sementes_pixels  = []   # coordenadas na imagem original
        self._canvas_img_scale = 1.0  # fator de escala imagem→canvas
        self._canvas_ph        = None # referência PhotoImage no canvas

        # Eventos de clique
        self._canvas_reg.bind("<Button-1>",   self._canvas_click)
        self._canvas_reg.bind("<Button-3>",   self._canvas_remove_last)

        # ── Watershed ──
        tk.Label(p, text="Watershed", fg=ACCENT2,
                 bg=BG_PANEL, font=FONT_LABEL).pack(**{**pad, "pady": (14, 4)})
        make_button(p, "▶ Executar Watershed", self._run_watershed,
                    color=ACCENT3).pack(padx=20, anchor="w")

        self._frm_watershed = tk.Frame(p, bg=BG_PANEL)
        self._frm_watershed.pack(padx=20, pady=4)

    # ── Canvas clicavel helpers ──────────────────────────────
    def _render_canvas_img(self):
        """
        Desenha a imagem atual no canvas de sementes, escalada para caber.
        Redesenha os marcadores de semente por cima.
        """
        if self._img is None:
            return
        from PIL import Image as PILImage
        img_h, img_w = self._img.shape[:2]
        scale = min(self._CANVAS_W / img_w, self._CANVAS_H / img_h)
        self._canvas_img_scale = scale
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)

        pil = PILImage.fromarray(self._img.astype(np.uint8))
        pil = pil.resize((disp_w, disp_h), PILImage.LANCZOS)
        self._canvas_ph = ImageTk.PhotoImage(pil)

        self._canvas_reg.config(width=disp_w, height=disp_h)
        self._canvas_reg.delete("all")
        self._canvas_reg.create_image(0, 0, anchor="nw", image=self._canvas_ph)
        self._canvas_img_offset = ((self._CANVAS_W - disp_w) // 2,
                                    (self._CANVAS_H - disp_h) // 2)
        # Redesenha marcadores
        for (sy, sx) in self._sementes_pixels:
            cx = int(sx * scale)
            cy = int(sy * scale)
            r = 5
            self._canvas_reg.create_oval(cx-r, cy-r, cx+r, cy+r,
                                          outline=ACCENT2, fill="", width=2)
            self._canvas_reg.create_oval(cx-1, cy-1, cx+1, cy+1,
                                          outline=ACCENT2, fill=ACCENT2)

    def _canvas_click(self, event):
        """Registra semente na coordenada clicada e atualiza display."""
        if self._img is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return
        scale = self._canvas_img_scale
        img_h, img_w = self._img.shape[:2]
        sx = int(event.x / scale)
        sy = int(event.y / scale)
        sx = max(0, min(sx, img_w - 1))
        sy = max(0, min(sy, img_h - 1))
        self._sementes_pixels.append((sy, sx))
        self._render_canvas_img()
        self._atualizar_lbl_sementes()

    def _canvas_remove_last(self, _event=None):
        """Remove a última semente com botão direito."""
        if self._sementes_pixels:
            self._sementes_pixels.pop()
            self._render_canvas_img()
            self._atualizar_lbl_sementes()

    def _limpar_sementes(self):
        self._sementes_pixels.clear()
        self._render_canvas_img()
        self._atualizar_lbl_sementes()

    def _atualizar_lbl_sementes(self):
        n = len(self._sementes_pixels)
        if n == 0:
            self._lbl_sementes_info.config(text="Sementes: nenhuma", fg=TEXT_DIM)
        else:
            coords = "  ".join(f"({sy},{sx})" for sy, sx in self._sementes_pixels)
            self._lbl_sementes_info.config(
                text=f"Sementes ({n}): {coords}", fg=ACCENT2)

    def _parse_sementes(self):
        """Retorna lista de sementes do canvas. Fallback para o campo de texto legado."""
        return list(self._sementes_pixels)

    def _run_regioes(self):
        if not self._check(): return
        if not self._sementes_pixels:
            messagebox.showwarning("Aviso", "Clique na imagem para definir pelo menos uma semente.")
            return
        try:
            T   = float(self._scale_reg.get())
            h, w = self._img.shape[:2]
            for sy, sx in self._sementes_pixels:
                if not (0 <= sy < h and 0 <= sx < w):
                    raise ValueError(f"Semente ({sy},{sx}) fora dos limites ({h}×{w}).")
            res = self._seg.region_growing(self._img, self._sementes_pixels, T)
            self._last_result = res
            # Exibe no card da direita
            show_array_in_label(res, self._lbl_reg_res, self._lbl_reg_info,
                                 self._CANVAS_W, self._CANVAS_H)
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _run_watershed(self):
        if not self._check(): return
        try:
            img_linhas = watershed(self._img)
            self._last_result = img_linhas
            self._show_row(self._frm_watershed, [
                (self._img,   "Original"),
                (img_linhas,  "Linhas de contenção"),
            ])
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def _save(self):
        if self._last_result is None:
            messagebox.showinfo("Info", "Execute alguma operação antes de salvar.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
        if path:
            Image.fromarray(self._last_result.astype(np.uint8)).save(path)
            messagebox.showinfo("Salvo", f"Imagem salva em:\n{path}")


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
            "Realce": AbaRealce(self._content),
            "Filtragem": AbaFiltragem(self._content, self),
            "Segmentacao": AbaSegmentacao(self._content),
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
            ("Realce", "Realce", ACCENT4),
            ("Filtragem", "Filtragem\nEspacial", ACCENT5),
            ("Segmentacao", "Segmentação", "#e8c468"),
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
