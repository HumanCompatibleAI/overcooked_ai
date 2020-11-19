import pygame
import os, copy
from overcooked_ai_py.utils import generate_temporary_file_path, classproperty
from overcooked_ai_py.static import GRAPHICS_DIR, FONTS_DIR
from overcooked_ai_py.mdp.layout_generator import EMPTY, COUNTER, ONION_DISPENSER, TOMATO_DISPENSER, POT, DISH_DISPENSER, SERVING_LOC
from overcooked_ai_py.visualization.visualization_utils import show_image_in_ipython
from overcooked_ai_py.visualization.pygame_utils import MultiFramePygameImage, run_static_resizeable_window, vstack_surfaces, scale_surface_by_factor, scale_surface_to_size
from overcooked_ai_py.mdp.actions import Direction

roboto_path = os.path.join(FONTS_DIR, "Roboto-Regular.ttf")


class StateVisualizer:
    TERRAINS_IMG = MultiFramePygameImage(os.path.join(GRAPHICS_DIR, 'terrain.png'), os.path.join(GRAPHICS_DIR, 'terrain.json'))
    OBJECTS_IMG = MultiFramePygameImage(os.path.join(GRAPHICS_DIR, 'objects.png'), os.path.join(GRAPHICS_DIR, 'objects.json'))
    SOUPS_IMG = MultiFramePygameImage(os.path.join(GRAPHICS_DIR, 'soups.png'), os.path.join(GRAPHICS_DIR, 'soups.json'))
    CHEFS_IMG = MultiFramePygameImage(os.path.join(GRAPHICS_DIR, 'chefs.png'), os.path.join(GRAPHICS_DIR, 'chefs.json'))
    UNSCALED_TILE_SIZE = 15
    DEFAULT_VALUES = {
        "height": None, # if None use grid_width - NOTE: can chop down hud if hud is wider than grid
        "width": None, # if None use (hud_height+grid_height)
        "tile_size": 75,
        "window_fps": 30,
        "player_colors": ['blue', 'green'],
        "is_rendering_hud": True,
        "hud_font_size": 10,
        "hud_font_path": roboto_path, 
        "hud_system_font_name": None, # if set to None use hud_font_path
        "hud_font_color": (255,255,255), # white
        "hud_data_default_key_order" : ["all_orders", "bonus_orders", "time_left", "score", "potential"],
        "hud_interline_size": 10,
        "hud_margin_bottom": 10,
        "hud_margin_top": 10,
        "hud_margin_left": 10,
        "hud_distance_between_orders": 5,
        "hud_order_size": 15,
        "is_rendering_cooking_timer": True,
        "show_timer_when_cooked": True,
        "cooking_timer_font_size": 20, # # if set to None use cooking_timer_font_path
        "cooking_timer_font_path": roboto_path, 
        "cooking_timer_system_font_name": None,
        "cooking_timer_font_color": (255, 0, 0), # red
        "grid": None,
        "background_color": (155, 101, 0) # color of empty counter
    }
    TILE_TO_FRAME_NAME = {
        EMPTY: "floor",
        COUNTER: "counter",
        ONION_DISPENSER: "onions",
        TOMATO_DISPENSER: "tomatoes",
        POT: "pot",
        DISH_DISPENSER: "dishes",
        SERVING_LOC: "serve"
    }
    
    def __init__(self, **kwargs):
        params = copy.deepcopy(self.DEFAULT_VALUES)
        params.update(kwargs)
        self.configure(**params)
        self.reload_fonts()

    def reload_fonts(self):
        pygame.font.init()
        if not hasattr(self, "_font"):
            self._fonts = {}
        # initializing fonts only if needed because it can take a quite long time, 
        #   see https://pygame.readthedocs.io/en/latest/4_text/text.html#initialize-a-font
        if self.is_rendering_hud:
            self.hud_font = self._init_font(self.hud_font_size, self.hud_font_path, self.hud_system_font_name)
        else:
            self.hud_font = None
            
        if self.is_rendering_cooking_timer:
            self.cooking_timer_font = self._init_font(self.cooking_timer_font_size, self.cooking_timer_font_path,
                                                     self.cooking_timer_system_font_name)
        else:
            self.cooking_timer_font = None
        
    @classmethod
    def configure_defaults(cls, **kwargs):
        cls._check_config_validity(kwargs)
        cls.DEFAULT_VALUES.update(copy.deepcopy(kwargs))

    def configure(self, **kwargs):
        StateVisualizer._check_config_validity(kwargs)
        for param_name, param_value in copy.deepcopy(kwargs).items():
            setattr(self, param_name, param_value)
        
    def display_rendered_state(self, state, hud_data=None, grid=None, img_path=None, ipython_display=False, window_display=False):
        """
        renders state as image
        img_path (str): if it is not None save image to specific path
        ipython_display (bool): if True render state in ipython cell, if img_path is None create file with randomized name in /tmp directory
        window_display (bool): if True render state into pygame window
        initial_scale_factor(float or int, int is recommended): how much increase size of the result image/window
        """
        assert window_display or img_path or ipython_display # without any of this param nothing happen when calling this function
        surface = self.render_state(state, grid, hud_data)

        if img_path is None and ipython_display:
            img_path = generate_temporary_file_path(prefix="/overcooked_visualized_state_", extension=".png")

        if img_path is not None:
            pygame.image.save(surface, img_path)

        if ipython_display:
            show_image_in_ipython(img_path)

        if window_display:
            run_static_resizeable_window(surface, self.window_fps)

        return img_path
    
    def render_state(self, state, grid, hud_data=None):
        """
        returns surface with rendered game state scaled by initial_scale_factor,
        decoupled from display_rendered_state function to make testing easier
        """
        pygame.init()
        grid = grid or self.grid
        assert grid
        grid_surface = pygame.surface.Surface(self._unscaled_grid_pixel_size(grid))
        self._render_grid(grid_surface, grid)
        self._render_players(grid_surface, state.players)
        self._render_objects(grid_surface, state.objects, grid)       
        if self.scale_by_factor != 1:
            grid_surface = scale_surface_by_factor(grid_surface, self.scale_by_factor)

        # render text after rescaling as text looks bad when is rendered small resolution and then rescalled to bigger one
        if self.is_rendering_cooking_timer:
            self._render_cooking_timers(grid_surface, state.objects, grid)
        
        if self.is_rendering_hud and hud_data:
            hud_width = self.width or grid_surface.get_width()
            hud_surface = pygame.surface.Surface((hud_width, self._calculate_hud_height(hud_data)))
            hud_surface.fill(self.background_color)
            self._render_hud_data(hud_surface, hud_data)
            rendered_surface = vstack_surfaces([hud_surface, grid_surface], self.background_color)
        else:
            hud_width = None
            rendered_surface = grid_surface
        
        result_surface_size = (self.width or rendered_surface.get_width(), self.height or rendered_surface.get_height())
        
        if result_surface_size != rendered_surface.get_size():
            result_surface = scale_surface_to_size(rendered_surface, result_surface_size, background_color=self.background_color)
        else:
            result_surface = rendered_surface

        return result_surface

    @property
    def scale_by_factor(self):
        return self.tile_size/StateVisualizer.UNSCALED_TILE_SIZE
    
    @property
    def hud_line_height(self):
        return self.hud_interline_size + self.hud_font_size

    @staticmethod
    def _check_config_validity(config):
        assert set(config.keys()).issubset(set(StateVisualizer.DEFAULT_VALUES.keys()))
        
    def _init_font(self, font_size, font_path=None, system_font_name=None):   
        if system_font_name:
            key = "%i-sys:%s" %(font_size, system_font_name)
            font = self._fonts.get(key) or pygame.font.SysFont(system_font_name, font_size)
        else:
            key = "%i-path:%s" %(font_size, font_path)
            font = self._fonts.get(key) or pygame.font.Font(font_path, font_size)
        self._fonts[key] = font
        return font

    def _unscaled_grid_pixel_size(self, grid):
        y_tiles = len(grid)
        x_tiles = len(grid[0])
        return (x_tiles * self.UNSCALED_TILE_SIZE, y_tiles * self.UNSCALED_TILE_SIZE)

    def _render_grid(self, surface, grid):
        for y_tile, row in enumerate(grid):
            for x_tile, tile in enumerate(row):
                self.TERRAINS_IMG.blit_on_surface(surface, self._position_in_unscaled_pixels((x_tile, y_tile)), 
                                             StateVisualizer.TILE_TO_FRAME_NAME[tile])

    def _position_in_unscaled_pixels(self, position):
        """
        get x and y coordinates in tiles, returns x and y coordinates in pixels
        """
        (x,y) = position
        return (self.UNSCALED_TILE_SIZE * x, self.UNSCALED_TILE_SIZE * y)
    
    def _position_in_scaled_pixels(self, position):
        """
        get x and y coordinates in tiles, returns x and y coordinates in pixels
        """
        (x,y) = position
        return (self.tile_size * x, self.tile_size * y)

    def _render_players(self, surface, players):
        def chef_frame_name(direction_name, held_object_name):
            frame_name = direction_name
            if held_object_name:
                frame_name += "-" + held_object_name
            return frame_name

        def hat_frame_name(direction_name, player_color_name):
            return "%s-%shat" %(direction_name, player_color_name)

        for player_num, player in enumerate(players):
            player_color_name = self.player_colors[player_num]
            direction_name = Direction.DIRECTION_TO_NAME[player.orientation]

            held_obj = player.held_object
            if held_obj is None:
                held_object_name = ""
            else:
                if held_obj.name == "soup":
                    if "onion" in held_obj.ingredients:
                        held_object_name = "soup-onion"
                    else:
                        held_object_name = "soup-tomato"
                else:
                    held_object_name = held_obj.name

            self.CHEFS_IMG.blit_on_surface(surface, self._position_in_unscaled_pixels(player.position), chef_frame_name(direction_name, held_object_name))
            self.CHEFS_IMG.blit_on_surface(surface, self._position_in_unscaled_pixels(player.position), hat_frame_name(direction_name, player_color_name))
    
    @staticmethod
    def _soup_frame_name(ingredients_names, status):
            num_onions = ingredients_names.count("onion")
            num_tomatoes = ingredients_names.count("tomato")
            return "soup_%s_tomato_%i_onion_%i" %(status, num_tomatoes, num_onions)

    def _render_objects(self, surface, objects, grid):
        def render_soup(surface, obj, grid):
            (x_pos, y_pos) = obj.position
            if grid[y_pos][x_pos] == POT:
                if obj.is_ready:
                    soup_status = "cooked"
                else:
                    soup_status = "idle"
            else: # grid[x][y] != POT
                soup_status = "done"
            frame_name = StateVisualizer._soup_frame_name(obj.ingredients, soup_status)
            self.SOUPS_IMG.blit_on_surface(surface, self._position_in_unscaled_pixels(obj.position), frame_name)

        for obj in objects.values():
            if obj.name == "soup":
                render_soup(surface, obj, grid)
            else:
                self.OBJECTS_IMG.blit_on_surface(surface, self._position_in_unscaled_pixels(obj.position), obj.name)            

    def _render_cooking_timers(self, surface, objects, grid):
        for key, obj in objects.items():
            (x_pos, y_pos) = obj.position
            if obj.name == "soup" and grid[y_pos][x_pos] == POT:
                if obj._cooking_tick != -1 and (obj._cooking_tick <= obj.cook_time or self.show_timer_when_cooked):
                    text_surface = self.cooking_timer_font.render(str(obj._cooking_tick), True, self.cooking_timer_font_color)
                    (tile_pos_x, tile_pos_y) = self._position_in_scaled_pixels(obj.position)

                    # calculate font position to be in center on x axis, and 0.9 from top on y axis
                    font_position = (tile_pos_x + int((self.tile_size - text_surface.get_width() ) * 0.5), 
                                 tile_pos_y + int((self.tile_size - text_surface.get_height() ) * 0.9))
                    surface.blit(text_surface, font_position)

    def _sorted_hud_items(self, hud_data):
        def default_order_then_alphabetic(item):
            key = item[0]
            try:
                i = self.hud_data_default_key_order.index(key)
            except:
                i = 99999
            return (i, key)
        return sorted(hud_data.items(), key=default_order_then_alphabetic)

    def _key_to_hud_text(self, key):
        return key.replace("_", " ").title() + ": "
    
    def _render_hud_data(self, surface, hud_data):
        def hud_text_position(line_num):
            return (self.hud_margin_left, self.hud_margin_top + self.hud_line_height * line_num)
        
        def hud_recipes_position(text_surface, text_surface_position):
            (text_surface_x, text_surface_y) = text_surface_position
            return (text_surface_x + text_surface.get_width(), text_surface_y)
        
        def get_hud_recipes_surface(orders_dicts):
            order_width = order_height = self.hud_order_size
            scaled_order_size = (order_width, order_width)
            orders_surface_height = order_height
            orders_surface_width = len(orders_dicts) * order_width + (len(orders_dicts) - 1) * self.hud_distance_between_orders
            unscaled_order_size = (self.UNSCALED_TILE_SIZE, self.UNSCALED_TILE_SIZE)
            
            recipes_surface = pygame.surface.Surface((orders_surface_width, orders_surface_height))
            recipes_surface.fill(self.background_color)
            next_surface_x = 0
            for order_dict in orders_dicts:
                frame_name = StateVisualizer._soup_frame_name(order_dict["ingredients"], "done")
                unscaled_order_surface = pygame.surface.Surface(unscaled_order_size)
                unscaled_order_surface.fill(self.background_color)
                self.SOUPS_IMG.blit_on_surface(unscaled_order_surface, (0,0), frame_name)
                if scaled_order_size == unscaled_order_size:
                    scaled_order_surface = unscaled_order_surface 
                else:
                    scaled_order_surface = pygame.transform.scale(unscaled_order_surface, (order_width, order_width))
                recipes_surface.blit(scaled_order_surface, (next_surface_x, 0))
                next_surface_x += order_width + self.hud_distance_between_orders
            return recipes_surface
        
        
        for hud_line_num, (key, value) in enumerate(self._sorted_hud_items(hud_data)):
            hud_text = self._key_to_hud_text(key)
            if key not in ["all_orders", "bonus_orders", "start_all_orders", "start_bonus_orders"]:
                hud_text += str(value)
            
            text_surface = self.hud_font.render(hud_text, True, self.hud_font_color)
            text_surface_position = hud_text_position(hud_line_num)
            surface.blit(text_surface, text_surface_position)

            if key in ["all_orders", "bonus_orders", "start_all_orders", "start_bonus_orders"] and value:
                recipes_surface_position = hud_recipes_position(text_surface, text_surface_position)
                recipes_surface = get_hud_recipes_surface(value)
                assert recipes_surface.get_width() + text_surface.get_width() <= surface.get_width()
                surface.blit(recipes_surface, recipes_surface_position)
                
    def _calculate_hud_height(self, hud_data):
        return self.hud_margin_top + len(hud_data) * self.hud_line_height + self.hud_margin_bottom
