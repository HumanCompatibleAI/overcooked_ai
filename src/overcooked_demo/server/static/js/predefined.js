// This is the javascript that defines transitions and interactions in the predefined.html page
// Persistent network connection that will be used to transmit real-time data
var socket = io();

var config;
var experimentParams = {
    layouts : ["cramped_room", "counter_circuit"],
    gameTime : 10,
    playerZero : "DummyAI"
};

var lobbyWaitTime = 300000;

/* * * * * * * * * * * * * 
 * Socket event handlers *
 * * * * * * * * * * * * */

window.intervalID = -1;
window.ellipses = -1;
window.lobbyTimeout = -1;

$(function() {
    $('#leave-btn').click(function () {
        socket.emit("leave",{});
        window.location.href = "/"
    });
});


socket.on('waiting', function(data) {
    // Show game lobby
    $('#game-over').hide();
    $("#overcooked").empty();
    $('#lobby').show();
    if (!data.in_game) {
        if (window.intervalID === -1) {
            // Occassionally ping server to try and join
            window.intervalID = setInterval(function() {
                socket.emit('join', {});
            }, 1000);
        }
    }
    if (window.lobbyTimeout === -1) {
        // Waiting animation
        window.ellipses = setInterval(function () {
            var e = $("#ellipses").text();
            $("#ellipses").text(".".repeat((e.length + 1) % 10));
        }, 500);
        // Timeout to leave lobby if no-one is found
        window.lobbyTimeout = setTimeout(function() {
            socket.emit('leave', {});
        }, config.lobbyWaitTime)
    }
});

socket.on('creation_failed', function(data) {
    // Tell user what went wrong
    let err = data['error']
    $("#overcooked").empty();
    $('#overcooked').append(`<h4>Sorry, game creation code failed with error: ${JSON.stringify(err)}</>`);
    $("error-exit").show();

    // Let parent window know error occurred
    window.top.postMessage({ name : "error"}, "*");
});

socket.on('start_game', function(data) {
    // Hide game-over and lobby, show game title header
    if (window.intervalID !== -1) {
        clearInterval(window.intervalID);
        window.intervalID = -1;
    }
    if (window.lobbyTimeout !== -1) {
        clearInterval(window.ellipses);
        clearTimeout(window.lobbyTimeout);
        window.lobbyTimeout = -1;
        window.ellipses = -1;
    }
    graphics_config = {
        container_id : "overcooked",
        start_info : data.start_info
    };
    $("#overcooked").empty();
    $('#game-over').hide();
    $('#lobby').hide();
    $('#reset-game').hide();
    $('#game-title').show();
    enable_key_listener();
    graphics_start(graphics_config);
});

socket.on('reset_game', function(data) {
    graphics_end();
    disable_key_listener();
    $("#overcooked").empty();
    $("#reset-game").show();
    setTimeout(function() {
        $("#reset-game").hide();
        graphics_config = {
            container_id : "overcooked",
            start_info : data.state
        };
        graphics_start(graphics_config);
        enable_key_listener();

        // Propogate game stats to parent window 
        window.top.postMessage({ name : "data", data : data.data, done : false}, "*");
    }, data.timeout);
});

socket.on('state_pong', function(data) {
    // Draw state update
    drawState(data['state']);
});

socket.on('end_game', function(data) {
    // Hide game data and display game-over html
    graphics_end();
    disable_key_listener();
    $('#game-title').hide();
    $('#game-over').show();
    $("#overcooked").empty();

    // Game ended unexpectedly
    if (data.status === 'inactive') {
        $("#error").show();
        $("#error-exit").show();
    }

    // Propogate game stats to parent window
    window.top.postMessage({ name : "data", data : data.data, done : true }, "*");
});

socket.on('end_lobby', function() {
    // Display join game timeout text
    $("#finding_partner").text(
        "We were unable to find you a partner."
    );
    $("#error-exit").show();

    // Stop trying to join
    clearInterval(window.intervalID);
    clearInterval(window.ellipses);
    window.intervalID = -1;

    // Let parent window know what happened
    window.top.postMessage({ name : "timeout" }, "*");
})


/* * * * * * * * * * * * * * 
 * Game Key Event Listener *
 * * * * * * * * * * * * * */

function enable_key_listener() {
    $(document).on('keydown', function(e) {
        let action = 'STAY'
        switch (e.which) {
            case 37: // left
                action = 'LEFT';
                break;

            case 38: // up
                action = 'UP';
                break;

            case 39: // right
                action = 'RIGHT';
                break;

            case 40: // down
                action = 'DOWN';
                break;

            case 32: //space
                action = 'SPACE';
                break;

            default: // exit this handler for other keys
                return; 
        }
        e.preventDefault();
        socket.emit('action', { 'action' : action });
    });
};

function disable_key_listener() {
    $(document).off('keydown');
};


/* * * * * * * * * * * * 
 * Game Initialization *
 * * * * * * * * * * * */

socket.on("connect", function() {
    // set configuration variables
    set_config();

    // Config for this specific game
    let uid = $('#uid').text();
    let params = JSON.parse(JSON.stringify(config.experimentParams));
    let data = {
        "params" : params,
        "game_name" : "overcooked"
    };

    // create (or join if it exists) new game
    socket.emit("join", data);
});


/* * * * * * * * * * *
 * Utility Functions *
 * * * * * * * * * * */

var arrToJSON = function(arr) {
    let retval = {}
    for (let i = 0; i < arr.length; i++) {
        elem = arr[i];
        key = elem['name'];
        value = elem['value'];
        retval[key] = value;
    }
    return retval;
};

var set_config = function() {
    config = JSON.parse($("#config").text());
}
