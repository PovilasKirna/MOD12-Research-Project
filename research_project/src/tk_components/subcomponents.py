import tkinter as tk
from tkinter import ttk
from awpy.types import PlayerInfo

from models.side_type import SideType

from enum import Enum

class PlayerInfoFrame(ttk.Frame):
    """A frame that displays information about a player."""
    player_name: str
    health_bar_canvas: tk.Canvas
    info_label: tk.Text

    def __init__(self, parent: ttk.Frame, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.health_bar_canvas = tk.Canvas(self, height=5)
        self.health_bar_canvas.pack(fill=tk.X)

        self.player_name = ''

        self.info_label = tk.Text(self, width=40, height=5)
        self.info_label.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.pack()
    
    def set_info(self, player_info: PlayerInfo):
        """Sets the player info to be displayed in the frame."""
        hp = player_info['hp']
        armor = player_info['armor']
        name = player_info['name']
        weapons = [w['weaponName'] for w in player_info['inventory'] or []]
        money = player_info['cash']
        has_bomb = player_info['hasBomb']
        has_defuse = player_info['hasDefuse']
        has_helmet = player_info['hasHelmet']

        self.info_label.config(state=tk.NORMAL)
        self.info_label.delete('1.0', tk.END)
        hp_status_info_string = f'{name} | HP: {hp}'
        if armor > 0:
            hp_status_info_string += f' | Armor {"(with Helmet)" if has_helmet else ""}: {armor}'
        self.info_label.insert(tk.END, f'{hp_status_info_string}\n')
        self.info_label.insert(tk.END, f'Weapons: {", ".join(weapons)}\n')
        self.info_label.insert(tk.END, f'Money: {money}\n')

        hp_bar_fill_color: str
        match SideType.from_str(player_info['side']):
            case SideType.T:
                self.info_label.insert(tk.END, f'Has Bomb: {has_bomb}\n')
                hp_bar_fill_color = 'goldenrod'
            case SideType.CT:
                self.info_label.insert(tk.END, f'Has Defuse: {has_defuse}\n')
                hp_bar_fill_color = 'lightblue'
        
        self.info_label.config(state=tk.DISABLED)

        # Update the health bar
        self.health_bar_canvas.delete('all')
        hp_bar_width = int(self.winfo_width() * (hp / 100))
        self.health_bar_canvas.create_rectangle(0, 0, hp_bar_width, 5, fill=hp_bar_fill_color)
        self.health_bar_canvas.update()

class RoutineMenuButtonNames(Enum):
    """The names of the buttons in the Routine menu enumified so we don't have to worry about the pitfalls of magic strings."""
    TOGGLE_ROUTINE_VISUALIZATION = 'Toggle Routine Visualization'
    SET_ROUTINE_LENGTH = 'Set Routine Length'

class HeatmapMenuButtonNames(Enum):
    """The names of the buttons in the Heatmap menu enumified so we don't have to worry about the pitfalls of magic strings."""
    SAVE_HEATMAP_FILE = 'Save Routine Heatmap File'
    LOAD_HEATMAP_FILE = 'Load Routine Heatmap File'
    
    GENERATE_POSITIONS_HEATMAP = 'Generate Positions Heatmap'
    GENERATE_ROUTINES_HEATMAP = 'Generate Routines Heatmap'
    GENERATE_ROUTINES_HEATMAP_FROM_DIRECTORY = 'Generate Routines Heatmap from Directory'

    VIEW_ROUTINE_HEATMAP_COMPOSITION_INFO = 'View List of Demos Used in Heatmap Creation'

    DRAW_POSITIONS_HEATMAP = 'Draw Positions Heatmap'
    DRAW_ROUTINES_HEATMAP_LINES = 'Draw Routines Heatmap (Lines)'
    DRAW_ROUTINES_HEATMAP_TILES = 'Draw Routines Heatmap (Tiles)'

    CLEAR_HEATMAP = 'Clear Heatmap'

class FrameWithScrollableInnerFrame(ttk.Frame):
    """A frame with a vertical scrollbar. Add widgets using the scrollable_frame as the parent."""
    scrollable_frame: ttk.Frame

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(
                scrollregion=canvas.bbox('all')
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
