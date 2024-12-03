/*

Added state potential to HUD

*/



// How long a graphics update should take in milliseconds
// Note that the server updates at 30 fps
var ANIMATION_DURATION = 50;

var DIRECTION_TO_NAME = {
    '0,-1': 'NORTH',
    '0,1': 'SOUTH',
    '1,0': 'EAST',
    '-1,0': 'WEST'
};

var scene_config = {
    player_colors : {0: 'blue', 1: 'green'},
    tileSize : 80,
    animation_duration : ANIMATION_DURATION,
    show_post_cook_time : false,
    cook_time : 20,
    assets_loc : "./static/assets/",
    hud_size : 150
};

var game_config = {
    type: Phaser.WEBGL,
    pixelArt: true,
    audio: {
        noAudio: true
    }
};

var graphics;

// Invoked at every state_pong event from server
function drawState(state) {
    // Try catch necessary because state pongs can arrive before graphics manager has finished initializing
    try {
        graphics.set_state(state);
    } catch {
        console.log("error updating state");
    }
};

// Invoked at 'start_game' event
function graphics_start(graphics_config) {
    graphics = new GraphicsManager(game_config, scene_config, graphics_config);
};

// Invoked at 'end_game' event
function graphics_end() {
    graphics.game.renderer.destroy();
    graphics.game.loop.stop();
    graphics.game.destroy();
}

class GraphicsManager {
    constructor(game_config, scene_config, graphics_config) {
        let start_info = graphics_config.start_info;
        scene_config.terrain = start_info.terrain;
        scene_config.start_state = start_info.state;
        game_config.scene = new OvercookedScene(scene_config);
        game_config.width = scene_config.tileSize*scene_config.terrain[0].length;
        game_config.height = scene_config.tileSize*scene_config.terrain.length  + scene_config.hud_size;
        game_config.parent = graphics_config.container_id;
        this.game = new Phaser.Game(game_config);
    }

    set_state(state) {
        this.game.scene.getScene('PlayGame').set_state(state);
    }
}

class OvercookedScene extends Phaser.Scene {
    constructor(config) {
        super({key: "PlayGame"});
        this.state = config.start_state.state;
        this.player_colors = config.player_colors;
        this.terrain = config.terrain;
        this.tileSize = config.tileSize;
        this.animation_duration = config.animation_duration;
        this.show_post_cook_time = config.show_post_cook_time;
        this.cook_time = config.cook_time;
        this.assets_loc = config.assets_loc;
        this.hud_size = config.hud_size
        this.hud_data = {
            potential : config.start_state.potential,
            score : config.start_state.score,
            time : config.start_state.time_left,
            bonus_orders : config.start_state.state.bonus_orders,
            all_orders : config.start_state.state.all_orders
        }
    }

    set_state(state) {
        this.hud_data.potential = state.potential;
        this.hud_data.score = state.score;
        this.hud_data.time = Math.round(state.time_left);
        this.hud_data.bonus_orders = state.state.bonus_orders;
        this.hud_data.all_orders = state.state.all_orders;
        this.state = state.state;
    }

    preload() {
        this.load.atlas("tiles",
            this.assets_loc + "terrain.png",
            this.assets_loc + "terrain.json");
        this.load.atlas("chefs",
            this.assets_loc + "chefs.png",
            this.assets_loc + "chefs.json");
        this.load.atlas("objects",
            this.assets_loc + "objects.png",
            this.assets_loc + "objects.json");
        this.load.multiatlas("soups",
            this.assets_loc + "soups.json",
            this.assets_loc)
    }

    create() {
        this.sprites = {};
        this.drawLevel();
        this._drawState(this.state, this.sprites);
    }

    update() {
        if (typeof(this.state) !== 'undefined') {
            this._drawState(this.state, this.sprites);
        }
        if (typeof(this.hud_data) !== 'undefined') {
            let { width, height } = this.game.canvas;
            let board_height = height - this.hud_size;
            this._drawHUD(this.hud_data, this.sprites, board_height);
        }
    }
    drawLevel() {
        // Fill canvas with white
        this.cameras.main.setBackgroundColor('#e6b453')

        //draw tiles
        let terrain_to_img = {
            ' ': 'floor.png',
            'X': 'counter.png',
            'P': 'pot.png',
            'O': 'onions.png',
            'T': 'tomatoes.png',
            'D': 'dishes.png',
            'S': 'serve.png'
        };
        let pos_dict = this.terrain;
        for (let row in pos_dict) {
            if (!pos_dict.hasOwnProperty(row)) {continue}
            for (let col = 0; col < pos_dict[row].length; col++) {
                let [x, y] = [col, row]
                let ttype = pos_dict[row][col];
                let tile = this.add.sprite(
                    this.tileSize * x,
                    this.tileSize * y,
                    "tiles",
                    terrain_to_img[ttype]
                );
                tile.setDisplaySize(this.tileSize, this.tileSize);
                tile.setOrigin(0);
            }
        }
    }
    _drawState (state, sprites) {
        sprites = typeof(sprites) === 'undefined' ? {} : sprites;

        //draw chefs
        sprites['chefs'] =
            typeof(sprites['chefs']) === 'undefined' ? {} : sprites['chefs'];
        for (let pi = 0; pi < state.players.length; pi++) {
            let chef = state.players[pi];
            let [x, y] = chef.position;
            let dir = DIRECTION_TO_NAME[chef.orientation];
            let held_obj = chef.held_object;
            if (typeof(held_obj) !== 'undefined' && held_obj !== null) {
                if (held_obj.name === 'soup') {
                    let ingredients = held_obj._ingredients.map(x => x['name']);
                    if (ingredients.includes('onion')) {
                        held_obj = "-soup-onion";
                    } else {
                        held_obj = "-soup-tomato";
                    }
                    
                }
                else {
                    held_obj = "-"+held_obj.name;
                }
            }
            else {
                held_obj = "";
            }
            if (typeof(sprites['chefs'][pi]) === 'undefined') {
                let chefsprite = this.add.sprite(
                    this.tileSize*x,
                    this.tileSize*y,
                    "chefs",
                    `${dir}${held_obj}.png`
                );
                chefsprite.setDisplaySize(this.tileSize, this.tileSize);
                chefsprite.depth = 1;
                chefsprite.setOrigin(0);
                let hatsprite = this.add.sprite(
                    this.tileSize*x,
                    this.tileSize*y,
                    "chefs",
                    `${dir}-${this.player_colors[pi]}hat.png`
                );
                hatsprite.setDisplaySize(this.tileSize, this.tileSize);
                hatsprite.depth = 2;
                hatsprite.setOrigin(0);
                sprites['chefs'][pi] = {chefsprite, hatsprite};
            }
            else {
                let chefsprite = sprites['chefs'][pi]['chefsprite'];
                let hatsprite = sprites['chefs'][pi]['hatsprite'];
                chefsprite.setFrame(`${dir}${held_obj}.png`);
                hatsprite.setFrame(`${dir}-${this.player_colors[pi]}hat.png`);
                this.tweens.add({
                    targets: [chefsprite, hatsprite],
                    x: this.tileSize*x,
                    y: this.tileSize*y,
                    duration: this.animation_duration,
                    ease: 'Linear',
                    onComplete: (tween, target, player) => {
                        target[0].setPosition(this.tileSize*x, this.tileSize*y);
                        //this.animating = false;
                    }
                })
            }
        }

        //draw environment objects
        if (typeof(sprites['objects']) !== 'undefined') {
            for (let objpos in sprites.objects) {
                let {objsprite, timesprite} = sprites.objects[objpos];
                objsprite.destroy();
                if (typeof(timesprite) !== 'undefined') {
                    timesprite.destroy();
                }
            }
        }
        sprites['objects'] = {};

        for (let objpos in state.objects) {
            if (!state.objects.hasOwnProperty(objpos)) { continue }
            let obj = state.objects[objpos];
            let [x, y] = obj.position;
            let terrain_type = this.terrain[y][x];
            let spriteframe;
            let soup_status;
            if ((obj.name === 'soup') && (terrain_type === 'P')) {
                let ingredients = obj._ingredients.map(x => x['name']);

                // select pot sprite
                if (!obj.is_ready) {
                    soup_status = "idle";
                }
                else {
                    soup_status = "cooked";
                }
                spriteframe = this._ingredientsToSpriteFrame(ingredients, soup_status);
                let objsprite = this.add.sprite(
                    this.tileSize*x,
                    this.tileSize*y,
                    "soups",
                    spriteframe
                );
                objsprite.setDisplaySize(this.tileSize, this.tileSize);
                objsprite.depth = 1;
                objsprite.setOrigin(0);
                let objs_here = {objsprite};

                // show time accordingly
                let show_time = true;
                if (obj._cooking_tick > obj.cook_time && !this.show_post_cook_time || obj._cooking_tick == -1) {
                    show_time = false;
                }
                if (show_time) {
                    let timesprite =  this.add.text(
                        this.tileSize*(x+.5),
                        this.tileSize*(y+.6),
                        String(obj._cooking_tick),
                        {
                            font: "25px Arial",
                            fill: "red",
                            align: "center",
                        }
                    );
                    timesprite.depth = 2;
                    objs_here['timesprite'] = timesprite;
                }

                sprites['objects'][objpos] = objs_here
            }
            else if (obj.name === 'soup') {
                let ingredients = obj._ingredients.map(x => x['name']);
                let soup_status = "done";
                spriteframe = this._ingredientsToSpriteFrame(ingredients, soup_status);
                let objsprite = this.add.sprite(
                    this.tileSize*x,
                    this.tileSize*y,
                    "soups",
                    spriteframe
                );
                objsprite.setDisplaySize(this.tileSize, this.tileSize);
                objsprite.depth = 1;
                objsprite.setOrigin(0);
                sprites['objects'][objpos] = {objsprite};
            }
            else {
                if (obj.name === 'onion') {
                    spriteframe = "onion.png";
                }
                else if (obj.name === 'tomato') {
                    spriteframe = "tomato.png";
                }
                else if (obj.name === 'dish') {
                    spriteframe = "dish.png";
                }
                let objsprite = this.add.sprite(
                    this.tileSize*x,
                    this.tileSize*y,
                    "objects",
                    spriteframe
                );
                objsprite.setDisplaySize(this.tileSize, this.tileSize);
                objsprite.depth = 1;
                objsprite.setOrigin(0);
                sprites['objects'][objpos] = {objsprite};
            }
        }        
    }

    _drawHUD(hud_data, sprites, board_height) {
        if (typeof(hud_data.all_orders) !== 'undefined') {
            this._drawAllOrders(hud_data.all_orders, sprites, board_height);
        }
        if (typeof(hud_data.bonus_orders) !== 'undefined') {
            this._drawBonusOrders(hud_data.bonus_orders, sprites, board_height);
        }
        if (typeof(hud_data.time) !== 'undefined') {
            this._drawTimeLeft(hud_data.time, sprites, board_height);
        }
        if (typeof(hud_data.score) !== 'undefined') {
            this._drawScore(hud_data.score, sprites, board_height);
        }
        if (typeof(hud_data.potential) !== 'undefined' && hud_data.potential !== null) {
            console.log(hud_data.potential)
            this._drawPotential(hud_data.potential, sprites, board_height);
        }
    }

    _drawBonusOrders(orders, sprites, board_height) {
        if (typeof(orders) !== 'undefined' && orders !== null) {
            let orders_str = "Bonus Orders: ";
            if (typeof(sprites['bonus_orders']) !== 'undefined') {
                // Clear existing orders
                sprites['bonus_orders']['orders'].forEach(element => {
                    element.destroy();
                });
                sprites['bonus_orders']['orders'] = [];

                // Update with new orders
                for (let i = 0; i < orders.length; i++) {
                    let spriteFrame = this._ingredientsToSpriteFrame(orders[i]['ingredients'], "done");
                    let orderSprite = this.add.sprite(
                        130 + 40 * i,
                        board_height + 40,
                        "soups",
                        spriteFrame
                    );
                    sprites['bonus_orders']['orders'].push(orderSprite);
                    orderSprite.setDisplaySize(60, 60);
                    orderSprite.setOrigin(0);
                    orderSprite.depth = 1;
                }
            }
            else {
                sprites['bonus_orders'] = {};
                sprites['bonus_orders']['str'] = this.add.text(
                    5, board_height + 60, orders_str,
                    {
                        font: "20px Arial",
                        fill: "red",
                        align: "left"
                    }
                )
                sprites['bonus_orders']['orders'] = []
            }
        }
    }

    _drawAllOrders(orders, sprites, board_height) {
        if (typeof(orders) !== 'undefined' && orders !== null) {
            let orders_str = "All Orders: ";
            if (typeof(sprites['all_orders']) !== 'undefined') {
                // Clear existing orders
                sprites['all_orders']['orders'].forEach(element => {
                    element.destroy();
                });
                sprites['all_orders']['orders'] = [];

                // Update with new orders
                for (let i = 0; i < orders.length; i++) {
                    let spriteFrame = this._ingredientsToSpriteFrame(orders[i]['ingredients'], "done");
                    let orderSprite = this.add.sprite(
                        90 + 40 * i,
                        board_height - 4,
                        "soups",
                        spriteFrame
                    );
                    sprites['all_orders']['orders'].push(orderSprite);
                    orderSprite.setDisplaySize(60, 60);
                    orderSprite.setOrigin(0);
                    orderSprite.depth = 1;
                }
            }
            else {
                sprites['all_orders'] = {};
                sprites['all_orders']['str'] = this.add.text(
                    5, board_height + 15, orders_str,
                    {
                        font: "20px Arial",
                        fill: "red",
                        align: "left"
                    }
                )
                sprites['all_orders']['orders'] = []
            }
        }
    }

    _drawScore(score, sprites, board_height) {
        score = "Score: "+score;
        if (typeof(sprites['score']) !== 'undefined') {
            sprites['score'].setText(score);
        }
        else {
            sprites['score'] = this.add.text(
                5, board_height + 90, score,
                {
                    font: "20px Arial",
                    fill: "red",
                    align: "left"
                }
            )
        }
    }

    _drawPotential(potential, sprites, board_height) {
        potential = "Potential: "+potential;
        if (typeof(sprites['potential']) !== 'undefined') {
            sprites['potential'].setText(potential);
        }
        else {
            sprites['potential'] = this.add.text(
                100, board_height + 90, potential,
                {
                    font: "20px Arial",
                    fill: "red",
                    align: "left"
                }
            )
        }
    }

    _drawTimeLeft(time_left, sprites, board_height) {
        time_left = "Time Left: "+time_left;
        if (typeof(sprites['time_left']) !== 'undefined') {
            sprites['time_left'].setText(time_left);
        }
        else {
            sprites['time_left'] = this.add.text(
                5, board_height + 115, time_left,
                {
                    font: "20px Arial",
                    fill: "red",
                    align: "left"
                }
            )
        }
    }

    _ingredientsToSpriteFrame(ingredients, status) {
        let num_tomatoes = ingredients.filter(x => x === 'tomato').length;
        let num_onions = ingredients.filter(x => x === 'onion').length;
        return `soup_${status}_tomato_${num_tomatoes}_onion_${num_onions}.png`
    }
}

