# Overcooked javascript app
<p align="center">
<img src="images/screenshot.png" width="350">
</p>
  
This is a javascript implementation of the Overcooked
MDP and game visualizer.

## Demo
To run a simple demo that plays a trajectory demonstrating the
transitions in the game:
```
$ open http://localhost:8123/demo.html; python2.7 -m SimpleHTTPServer 8123
```

Or if you have npm installed:
```
$ npm run demo
```

## Development
Set up the package with `npm install`.

Run tests with `npm run test`. Testing scripts use `jest`, which exposes a `window` object, and so
`npm run build-window` should be run before running modified tests.

`overcooked-window.js` is used for the demo and testing.