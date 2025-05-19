import tkinter as tk

from tk_components.components import MainApplication

# Time to design a GUI? # TODO.

def center_root(root: tk.Tk):
    """Centers the root window on the screen. See: https://stackoverflow.com/a/10018670"""
    root.update_idletasks()
    width = root.winfo_width()
    frm_width = root.winfo_rootx() - root.winfo_x()
    win_width = width + 2 * frm_width
    height = root.winfo_height()
    titlebar_height = root.winfo_rooty() - root.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = root.winfo_screenwidth() // 2 - win_width // 2
    y = root.winfo_screenheight() // 2 - win_height // 2
    root.geometry(f'{width}x{height}+{x}+{y}')
    root.deiconify()

def main():
    root = tk.Tk()
    root.attributes('-alpha', 0.0) # Hide the window in case centering it makes it flicker around
    root.title("CS:GO Demo Visualizer")
    root.geometry("1920x1080")
    MainApplication(root).pack(side='top', fill='both', expand=True)

    # Exit the application when the window is closed
    # This is required because, for some reason, the initial instantiation of the FigureCanvasTkAgg prevents the script from terminating even after the window is closed.
    # If this canvas is ever destroyed and re-created (e.g. when a new file is loaded), the script will terminate as expected.
    # To prevent issues in that initial case, this manual exit bind is added.
    # See: https://stackoverflow.com/a/67663998
    root.protocol("WM_DELETE_WINDOW", lambda: MainApplication(root).exit())

    center_root(root)
    root.attributes('-alpha', 1.0) # The window is done moving around, so show it now
    root.mainloop()

if __name__ == '__main__':
    main()