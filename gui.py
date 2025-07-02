import tkinter as tk
from tkinter import filedialog, messagebox
from prep_function import (
    load_muse_data, base_filtering,
    median_filter_artifact_removal, dynamic_threshold_artifact_removal,
    auto_artifact_rejection, run_ica, annotate_ica_artifacts
)
from plot import plot_single_channel_with_annotations

def select_file():
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filepath:
        path_var.set(filepath)

def run_pipeline():
    filepath = path_var.get()
    if not filepath:
        messagebox.showwarning("לא נבחר קובץ", "אנא בחרו קובץ CSV")
        return

    try:
        raw = load_muse_data(filepath)

        if filter_var.get():
            raw = base_filtering(raw)
        if median_var.get():
            raw = median_filter_artifact_removal(raw)
        if mad_var.get():
            raw = dynamic_threshold_artifact_removal(raw)
        if amplitude_var.get():
            raw = auto_artifact_rejection(raw)
        if ica_var.get():
            ica = run_ica(raw)
            ica.exclude = [0]
            raw = annotate_ica_artifacts(raw, ica)
            ica.apply(raw)

        plot_single_channel_with_annotations(raw)
        messagebox.showinfo("הצלחה", "הניקוי הסתיים והגרף הוצג.")

    except Exception as e:
        messagebox.showerror("שגיאה", str(e))


# === GUI ===
root = tk.Tk()
root.title("EEG Cleaning GUI - בחירת שלבים")
root.geometry("500x400")

# קובץ
path_var = tk.StringVar()
tk.Label(root, text="קובץ CSV:").pack()
tk.Entry(root, textvariable=path_var, width=60).pack(pady=5)
tk.Button(root, text="בחירת קובץ", command=select_file).pack()

filter_var = tk.BooleanVar(value=True)
median_var = tk.BooleanVar()
mad_var = tk.BooleanVar()
amplitude_var = tk.BooleanVar()
ica_var = tk.BooleanVar()

tk.Checkbutton(root, text="פילטר בסיסי (Base Filtering)", variable=filter_var).pack(anchor="w")
tk.Checkbutton(root, text="סינון עם חציון (Median Filter)", variable=median_var).pack(anchor="w")
tk.Checkbutton(root, text="סינון MAD (Threshold)", variable=mad_var).pack(anchor="w")
tk.Checkbutton(root, text="סינון לפי אמפליטודה", variable=amplitude_var).pack(anchor="w")
tk.Checkbutton(root, text="ICA להסרת רכיבים לא מוחיים", variable=ica_var).pack(anchor="w")

tk.Button(root, text="הרצת ניקוי והצגת גרף", command=run_pipeline).pack(pady=20)

def main():
    root.mainloop()

if __name__ == "__main__":
    main()