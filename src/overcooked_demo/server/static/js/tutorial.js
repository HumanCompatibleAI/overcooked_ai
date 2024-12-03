// Persistent network connection that will be used to transmit real-time data
var socket = io();



var config;

var tutorial_instructions = () => [
    `
    <p>Mechanic: <b>Delivery</b></p>
    <p>Your goal here is to cook and deliver soups in order to earn reward. Notice how your partner is busily churning out soups</p>
    <p>See if you can copy his actions in order to cook and deliver the appropriate soup</p>
    <p><b>Note</b>: only recipes in the <b>All Orders</b> field will earn reward. Thus, you must cook a soup with <b>exactly</b> 3 onions</p>
    <p><b>You will advance only when you have delivered the appropriate soup</b></p>
    <p>Good luck!</p>
    <br></br>
    `,
    `
    <p>Mechanic: <b>All Orders</b></p>
    <p>Oh no! Your partner has made a grave mistake! He has mistakingly placed two onions in the pot</p>
    <p>This is an issue because no recipe on the <b>All Orders</b> list can started with 2 onions</p>
    <p>See if you can remedy the situation and cook a recipe that is indeed valid</p>
    <p><b>You will advance only when you have delivered a valid soup</b></p>
    <p>Good Luck!</p>
    <br></br>
    `,
    `
    <p>Mechanic: <b>Scoring</b></p>
    <p>Your partner is again back again busily busting out onion soups, except this time, we have a problem...</p>
    <p>The customers in this restaurant are super picky! They will only eat a soup that is worth exactly <b>${config['tutorialParams']['phaseTwoScore']} points</b></p>
    <p>Your goal here is to cooperate with your partner and cook a soup to satisfy the fastidious foodies</p>
    <p><b>You will advance only when you deliver a soup worth exactly ${config['tutorialParams']['phaseTwoScore']} points</b></p>
    <br></br>
    `,
    `
    <p>One last mechanic: <b>Bonus Orders</b></p> 
    <p>In addition to the <b>All Orders</b> list, recipes in <b>Bonus Orders</b> are worth extra points!</p>
    <p>Your goal here is to cook and deliver a <b>bonus order</b></p>
    <p>Even though you can earn reward for other orders, <b>you will advance only when you have delivered a bonus order</b></p>
    <p>Good Luck!</p>
    <br></br>
    `
];

var tutorial_hints = () => [
    `
    <p>
        You can move up, down, left, and right using
        the <b>arrow keys</b>, and interact with objects
        using the <b>spacebar</b>.
      </p>
      <p>
        You can interact with objects by facing them and pressing
        <b>spacebar</b>. Here are some examples:
        <ul>
          <li>You can pick up ingredients (onions or tomatoes) by facing
            the ingredient area and pressing <b>spacebar</b>.</li>
          <li>If you are holding an ingredient, are facing an empty counter,
            and press <b>spacebar</b>, you put the ingredient on the counter.</li>
          <li>If you are holding an ingredient, are facing a pot that is not full,
            and press <b>spacebar</b>, you will put the ingredient in the pot.</li>
          <li>If you are facing a pot that is non-empty, are currently holding nothing, and 
            and press <b>spacebar</b>, you will begin cooking a soup.</li>
        </ul>
      </p>
    `,
    `
    <p>You cannot remove ingredients from the pot. You can, however, cook any soup you like, even if it's not in <b>All Orders</b>...</p>
    `,
    `
    <p>Each onion is worth ${config['onion_value']} points and each tomato is worth ${config['tomato_value']} points<p>
    `,
    `
    <p>The bonus order here is <b>1 onion 2 tomatoes<b>. This could be determined by referring to the soup legend </p>
    `
]

var curr_tutorial_phase;

// Read in game config provided by server
$(function() {
    config = JSON.parse($('#config').text());
    tutorial_instructions = tutorial_instructions();
    tutorial_hints = tutorial_hints();
    $('#quit').show();
});

/* * * * * * * * * * * * * * * * 
 * Button click event handlers *
 * * * * * * * * * * * * * * * */

$(function() {
    $('#try-again').click(function () {
        data = {
            "params" : config['tutorialParams'],
            "game_name" : "tutorial"
        };
        socket.emit("join", data);
        $('try-again').attr("disable", true);
    });
});

$(function() {
    $('#show-hint').click(function() {
        let text = $(this).text();
        let new_text = text === "Show Hint" ? "Hide Hint" : "Show Hint";
        $('#hint-wrapper').toggle();
        $(this).text(new_text);
    });
});

$(function() {
    $('#quit').click(function() {
        socket.emit("leave", {});
        $('quit').attr("disable", true);
        window.location.href = "./";
    });
});

$(function() {
    $('#finish').click(function() {
        $('finish').attr("disable", true);
        window.location.href = "./";
    });
});



/* * * * * * * * * * * * * 
 * Socket event handlers *
 * * * * * * * * * * * * */

socket.on('creation_failed', function(data) {
    // Tell user what went wrong
    let err = data['error']
    $("#overcooked").empty();
    $('#overcooked').append(`<h4>Sorry, tutorial creation code failed with error: ${JSON.stringify(err)}</>`);
    $('#try-again').show();
    $('#try-again').attr("disabled", false);
});

socket.on('start_game', function(data) {
    curr_tutorial_phase = 0;
    graphics_config = {
        container_id : "overcooked",
        start_info : data.start_info
    };
    $("#overcooked").empty();
    $('#game-over').hide();
    $('#try-again').hide();
    $('#try-again').attr('disabled', true)
    $('#hint-wrapper').hide();
    $('#show-hint').text('Show Hint');
    $('#game-title').text(`Tutorial in Progress, Phase ${curr_tutorial_phase}/${tutorial_instructions.length}`);
    $('#game-title').show();
    $('#tutorial-instructions').append(tutorial_instructions[curr_tutorial_phase]);
    $('#instructions-wrapper').show();
    $('#hint').append(tutorial_hints[curr_tutorial_phase]);
    enable_key_listener();
    graphics_start(graphics_config);
});

socket.on('reset_game', function(data) {
    curr_tutorial_phase++;
    graphics_end();
    disable_key_listener();
    $("#overcooked").empty();
    $('#tutorial-instructions').empty();
    $('#hint').empty();
    $("#tutorial-instructions").append(tutorial_instructions[curr_tutorial_phase]);
    $("#hint").append(tutorial_hints[curr_tutorial_phase]);
    $('#game-title').text(`Tutorial in Progress, Phase ${curr_tutorial_phase + 1}/${tutorial_instructions.length}`);
    
    let button_pressed = $('#show-hint').text() === 'Hide Hint';
    if (button_pressed) {
        $('#show-hint').click();
    }
    graphics_config = {
        container_id : "overcooked",
        start_info : data.state
    };
    graphics_start(graphics_config);
    enable_key_listener();
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
    $('#instructions-wrapper').hide();
    $('#hint-wrapper').hide();
    $('#show-hint').hide();
    $('#game-over').show();
    $('#quit').hide();
    
    if (data.status === 'inactive') {
        // Game ended unexpectedly
        $('#error-exit').show();
        // Propogate game stats to parent window 
        window.top.postMessage({ name : "error" }, "*");
    } else {
        // Propogate game stats to parent window 
        window.top.postMessage({ name : "tutorial-done" }, "*");
    }

    $('#finish').show();
});


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
    // Config for this specific game
    let data = {
        "params" : config['tutorialParams'],
        "game_name" : "tutorial"
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