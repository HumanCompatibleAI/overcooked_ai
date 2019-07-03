import Phaser from 'phaser'
import * as OvercookedMDP from "./mdp.es6"

let Direction = OvercookedMDP.Direction;

export class OvercookedGame {
    constructor ({
        start_grid,

        container_id,

        tileSize = 128,
        gameWidth = tileSize*start_grid[0].length,
        gameHeight = tileSize*start_grid.length,

        ANIMATION_DURATION = 500,
        TIMESTEP_DURATION = 600,
        player_colors = {0: 'green', 1: 'blue'},
        assets_loc = "./assets/",
        show_post_cook_time = false,

        COOK_TIME = 2,
        explosion_time = Number.MAX_SAFE_INTEGER,
        DELIVERY_REWARD = OvercookedMDP.OvercookedGridworld.DELIVERY_REWARD,
        always_serve = false
    }){
        this.gameWidth = gameWidth;
        this.gameHeight = gameHeight;
        this.container_id = container_id;
        let params = {COOK_TIME, explosion_time, DELIVERY_REWARD, always_serve};
        this.mdp = OvercookedMDP.OvercookedGridworld.from_grid(start_grid, params);
        this.state = this.mdp.get_start_state();
        this.joint_action = [OvercookedMDP.Direction.STAY, OvercookedMDP.Direction.STAY];
        this.player_colors = player_colors;
        this.assets_loc = assets_loc;
        this.show_post_cook_time = show_post_cook_time;

        let gameparent = this;
        this.scene = new Phaser.Class({
            gameparent,
            Extends: Phaser.Scene,
            initialize: function() {
                Phaser.Scene.call(this, {key: "PlayGame"})
            },
            preload: function () {
                this.load.atlas("tiles",
                    this.gameparent.assets_loc+"terrain.png",
                    this.gameparent.assets_loc+"terrain.json");
                this.load.atlas("chefs",
                    this.gameparent.assets_loc+"chefs.png",
                    this.gameparent.assets_loc+"chefs.json");
                this.load.atlas("objects",
                    this.gameparent.assets_loc+"objects.png",
                    this.gameparent.assets_loc+"objects.json");
            },
            create: function () {
                // this.gameparent = gameparent;
                this.mdp = gameparent.mdp;
                this.sprites = {};
                this.drawLevel();
                this._drawState(gameparent.state, this.sprites);
                // this.cursors = this.input.keyboard.createCursorKeys(); //this messes with the keys a lot
                // this.player.can_take_input = true;
                // this.animating_transition = false;
            },
            drawLevel: function() {
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
                let pos_dict = this.mdp._get_terrain_type_pos_dict();
                for (let ttype in pos_dict) {
                    if (!pos_dict.hasOwnProperty(ttype)) {continue}
                    for (let i = 0; i < pos_dict[ttype].length; i++) {
                        let [x, y] = pos_dict[ttype][i];
                        let tile = this.add.sprite(
                            tileSize * x,
                            tileSize * y,
                            "tiles",
                            terrain_to_img[ttype]
                        );
                        tile.setDisplaySize(tileSize, tileSize);
                        tile.setOrigin(0);
                    }
                }

            },
            _drawState: function (state, sprites) {
                sprites = typeof(sprites) === 'undefined' ? {} : sprites;

                //draw chefs
                sprites['chefs'] =
                    typeof(sprites['chefs']) === 'undefined' ? {} : sprites['chefs'];
                for (let pi = 0; pi < state.players.length; pi++) {
                    let chef = state.players[pi];
                    let [x, y] = chef.position;
                    let dir = Direction.DIRECTION_TO_NAME[chef.orientation];
                    let held_obj = chef.held_object;
                    if (typeof(held_obj) !== 'undefined') {
                        if (held_obj.name === 'soup') {
                            held_obj = "-soup-"+held_obj.state[0];
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
                            tileSize*x,
                            tileSize*y,
                            "chefs",
                            `${dir}${held_obj}.png`
                        );
                        chefsprite.setDisplaySize(tileSize, tileSize);
                        chefsprite.depth = 1;
                        chefsprite.setOrigin(0);
                        let hatsprite = this.add.sprite(
                            tileSize*x,
                            tileSize*y,
                            "chefs",
                            `${dir}-${this.gameparent.player_colors[pi]}hat.png`
                        );
                        hatsprite.setDisplaySize(tileSize, tileSize);
                        hatsprite.depth = 2;
                        hatsprite.setOrigin(0);
                        sprites['chefs'][pi] = {chefsprite, hatsprite};
                    }
                    else {
                        let chefsprite = sprites['chefs'][pi]['chefsprite'];
                        let hatsprite = sprites['chefs'][pi]['hatsprite'];
                        chefsprite.setFrame(`${dir}${held_obj}.png`);
                        hatsprite.setFrame(`${dir}-${this.gameparent.player_colors[pi]}hat.png`);
                        this.tweens.add({
                            targets: [chefsprite, hatsprite],
                            x: tileSize*x,
                            y: tileSize*y,
                            duration: ANIMATION_DURATION,
                            ease: 'Linear',
                            onComplete: (tween, target, player) => {
                                target[0].setPosition(tileSize*x, tileSize*y);
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
                    let terrain_type = this.mdp.get_terrain_type_at(obj.position);
                    let spriteframe, souptype, n_ingredients;
                    let cooktime = "";
                    if ((obj.name === 'soup') && (terrain_type === 'P')) {
                        [souptype, n_ingredients, cooktime] = obj.state;

                        // select pot sprite
                        if (cooktime <= this.mdp.COOK_TIME) {
                            spriteframe =
                                `soup-${souptype}-${n_ingredients}-cooking.png`;
                        }
                        else if (cooktime >= this.mdp.explosion_time) {
                            spriteframe = 'pot-explosion.png';
                        }
                        else {
                            spriteframe = `soup-${souptype}-cooked.png`;
                        }
                        let objsprite = this.add.sprite(
                            tileSize*x,
                            tileSize*y,
                            "objects",
                            spriteframe
                        );
                        objsprite.setDisplaySize(tileSize, tileSize);
                        objsprite.depth = 1;
                        objsprite.setOrigin(0);
                        let objs_here = {objsprite};

                        // show time accordingly
                        let show_time = true;
                        if ((cooktime > this.mdp.COOK_TIME) && !this.show_post_cook_time) {
                            show_time = false;
                        }
                        if (show_time) {
                            let timesprite =  this.add.text(
                                tileSize*(x+.5),
                                tileSize*(y+.6),
                                String(cooktime),
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
                        [souptype, n_ingredients, cooktime] = obj.state;
                        spriteframe = `soup-${souptype}-dish.png`;
                        let objsprite = this.add.sprite(
                            tileSize*x,
                            tileSize*y,
                            "objects",
                            spriteframe
                        );
                        objsprite.setDisplaySize(tileSize, tileSize);
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
                            tileSize*x,
                            tileSize*y,
                            "objects",
                            spriteframe
                        );
                        objsprite.setDisplaySize(tileSize, tileSize);
                        objsprite.depth = 1;
                        objsprite.setOrigin(0);
                        sprites['objects'][objpos] = {objsprite};
                    }
                }

                //draw order list
                let order_list = "Orders: "+state.order_list.join(", ");
                if (typeof(sprites['order_list']) !== 'undefined') {
                    sprites['order_list'].setText(order_list);
                }
                else {
                    sprites['order_list'] = this.add.text(
                        5, 5, order_list,
                        {
                            font: "20px Arial",
                            fill: "yellow",
                            align: "left"
                        }
                    )
                }
            },
            _drawScore: function(score, sprites) {
                score = "Score: "+score;
                if (typeof(sprites['score']) !== 'undefined') {
                    sprites['score'].setText(score);
                }
                else {
                    sprites['score'] = this.add.text(
                        5, 25, score,
                        {
                            font: "20px Arial",
                            fill: "yellow",
                            align: "left"
                        }
                    )
                }
            },
            _drawTimeLeft: function(time_left, sprites) {
                time_left = "Time Left: "+time_left;
                if (typeof(sprites['time_left']) !== 'undefined') {
                    sprites['time_left'].setText(time_left);
                }
                else {
                    sprites['time_left'] = this.add.text(
                        5, 45, time_left,
                        {
                            font: "20px Arial",
                            fill: "yellow",
                            align: "left"
                        }
                    )
                }
            },
            update: function() {
                // let state, score_;
                // let redraw = false;
                if (typeof(this.gameparent.state_to_draw) !== 'undefined') {
                    let state = this.gameparent.state_to_draw;
                    delete this.gameparent.state_to_draw;
                    // redraw = true;
                    this._drawState(state, this.sprites);
                }
                if (typeof(this.gameparent.score_to_draw) !== 'undefined') {
                    let score = this.gameparent.score_to_draw;
                    delete this.gameparent.score_to_draw;
                    this._drawScore(score, this.sprites);
                }
                if (typeof(this.gameparent.time_left) !== 'undefined') {
                    let time_left = this.gameparent.time_left;
                    delete this.gameparent.time_left;
                    this._drawTimeLeft(time_left, this.sprites);
                }
                // if (!redraw) {
                //     return
                // }

            }
        });
    }

    init () {
        let gameConfig = {
            type: Phaser.WEBGL,
            width: this.gameWidth,
            height: this.gameHeight,
            scene: [this.scene],
            parent: this.container_id,
            pixelArt: true,
            audio: {
                noAudio: true
            }
        };
        this.game = new Phaser.Game(gameConfig);
    }

    drawState(state) {
        this.state_to_draw = state;
    }

    setAction(player_index, action) {

    }

    drawScore(score) {
        this.score_to_draw = String(score);
    }

    drawTimeLeft(time_left) {
        this.time_left = String(time_left);
    }

    close (msg) {
        this.game.renderer.destroy();
        this.game.loop.stop();
        // this.game.canvas.remove();
        this.game.destroy();
    }

}