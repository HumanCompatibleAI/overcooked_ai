var container_id;

const noop = () => {};

function drawState(state) {
    $(`#${container_id}`).empty();
    $(`#${container_id}`).append(`<h4>Current Game State: ${JSON.stringify(state)}</>`);
};

function graphics_start(config) {
    container_id = config.container_id;
};

function graphics_end() {
    noop();
}