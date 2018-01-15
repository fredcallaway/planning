// Generated by CoffeeScript 2.0.2
/*
jspsych-mouselab-mdp.coffee
Fred Callaway

https://github.com/fredcallaway/Mouselab-MDP
*/
var D, SCORE, TIME_LEFT, WATCH, mdp;

// coffeelint: disable=max_line_length
mdp = void 0;

D = void 0;

WATCH = {};

SCORE = 0;

TIME_LEFT = void 0;

jsPsych.plugins['mouselab-mdp'] = (function() {
  var Arrow, Edge, KEYS, LOG_DEBUG, LOG_INFO, MouselabMDP, NULL, PRINT, RIGHT_MESSAGE, SIZE, State, TRIAL_INDEX, Text, angle, checkObj, dist, plugin, polarMove, redGreen, removePrivate, round;
  PRINT = function(...args) {
    return console.log(...args);
  };
  NULL = function(...args) {
    return null;
  };
  LOG_INFO = PRINT;
  LOG_DEBUG = NULL;
  // a scaling parameter, determines size of drawn objects
  SIZE = void 0;
  TRIAL_INDEX = 0;
  fabric.Object.prototype.originX = fabric.Object.prototype.originY = 'center';
  fabric.Object.prototype.selectable = false;
  fabric.Object.prototype.hoverCursor = 'plain';
  // =========================== #
  // ========= Helpers ========= #
  // =========================== #
  removePrivate = function(obj) {
    return _.pick(obj, (function(v, k, o) {
      return !k.startsWith('_');
    }));
  };
  angle = function(x1, y1, x2, y2) {
    var ang, x, y;
    x = x2 - x1;
    y = y2 - y1;
    if (x === 0) {
      ang = y === 0 ? 0 : y > 0 ? Math.PI / 2 : Math.PI * 3 / 2;
    } else if (y === 0) {
      ang = x > 0 ? 0 : Math.PI;
    } else {
      ang = x < 0 ? Math.atan(y / x) + Math.PI : y < 0 ? Math.atan(y / x) + 2 * Math.PI : Math.atan(y / x);
    }
    return ang + Math.PI / 2;
  };
  polarMove = function(x, y, ang, dist) {
    x += dist * Math.sin(ang);
    y -= dist * Math.cos(ang);
    return [x, y];
  };
  dist = function(o1, o2) {
    return Math.pow(Math.pow(o1.left - o2.left, 2) + Math.pow(o1.top - o2.top, 2), 0.5);
  };
  redGreen = function(val) {
    if (val > 0) {
      return '#080';
    } else if (val < 0) {
      return '#b00';
    } else {
      return '#666';
    }
  };
  round = function(x) {
    return (Math.round(x * 100)) / 100;
  };
  checkObj = function(obj, keys) {
    var i, k, len;
    if (keys == null) {
      keys = Object.keys(obj);
    }
    for (i = 0, len = keys.length; i < len; i++) {
      k = keys[i];
      if (obj[k] === void 0) {
        console.log('Bad Object: ', obj);
        throw new Error(`${k} is undefined`);
      }
    }
    return obj;
  };
  KEYS = mapObject({
    up: 'uparrow',
    down: 'downarrow',
    right: 'rightarrow',
    left: 'leftarrow',
    simulate: 'space'
  }, jsPsych.pluginAPI.convertKeyCharacterToKeyCode);
  RIGHT_MESSAGE = '\xa0'.repeat(8) + 'Score: <span id=mouselab-score/>';
  // =============================== #
  // ========= MouselabMDP ========= #
  // =============================== #
  MouselabMDP = class MouselabMDP {
    constructor(config) {
      var blockName, centerMessage, leftMessage, lowerMessage, prompt, rightMessage, size;
      this.startTimer = this.startTimer.bind(this);
      // ---------- Responding to user input ---------- #

      // Called when a valid action is initiated via a key press.
      this.handleKey = this.handleKey.bind(this);
      this.startSimulationMode = this.startSimulationMode.bind(this);
      this.endSimulationMode = this.endSimulationMode.bind(this);
      this.getOutcome = this.getOutcome.bind(this);
      this.getReward = this.getReward.bind(this);
      this.move = this.move.bind(this);
      this.clickState = this.clickState.bind(this);
      this.mouseoverState = this.mouseoverState.bind(this);
      this.mouseoutState = this.mouseoutState.bind(this);
      this.clickEdge = this.clickEdge.bind(this);
      this.mouseoverEdge = this.mouseoverEdge.bind(this);
      this.mouseoutEdge = this.mouseoutEdge.bind(this);
      this.getEdgeLabel = this.getEdgeLabel.bind(this);
      this.recordQuery = this.recordQuery.bind(this);
      // ---------- Updating state ---------- #

      // Called when the player arrives in a new state.
      this.arrive = this.arrive.bind(this);
      this.addScore = this.addScore.bind(this);
      this.resetScore = this.resetScore.bind(this);
      this.drawScore = this.drawScore.bind(this);
      this.spendEnergy = this.spendEnergy.bind(this);
      // $('#mouselab-').css 'color', redGreen SCORE

      // ---------- Starting the trial ---------- #
      this.run = this.run.bind(this);
      // Draw object on the canvas.
      this.draw = this.draw.bind(this);
      // Draws the player image.
      this.initPlayer = this.initPlayer.bind(this);
      // Constructs the visual display.
      this.buildMap = this.buildMap.bind(this);
      // ---------- ENDING THE TRIAL ---------- #

      // Creates a button allowing user to move to the next trial.
      this.endTrial = this.endTrial.bind(this);
      this.checkFinished = this.checkFinished.bind(this);
      // @transition=null  # function `(s0, a, s1, r) -> null` called after each transition
      
      // leftMessage="Round: #{TRIAL_INDEX}/#{N_TRIAL}"
      ({display: this.display, graph: this.graph, layout: this.layout, initial: this.initial, stateLabels: this.stateLabels = null, stateDisplay: this.stateDisplay = 'never', stateClickCost: this.stateClickCost = 0, edgeLabels: this.edgeLabels = 'reward', edgeDisplay: this.edgeDisplay = 'always', edgeClickCost: this.edgeClickCost = 0, stateRewards: this.stateRewards = null, clickDelay: this.clickDelay = 0, moveDelay: this.moveDelay = 1000, clickEnergy: this.clickEnergy = 0, moveEnergy: this.moveEnergy = 0, allowSimulation: this.allowSimulation = true, revealRewards: this.revealRewards = true, training: this.training = false, special: this.special = '', timeLimit: this.timeLimit = null, energyLimit: this.energyLimit = null, keys: this.keys = KEYS, trialIndex: this.trialIndex = TRIAL_INDEX, playerImage: this.playerImage = 'static/images/plane.png', size = 80, blockName = 'none', prompt = '&nbsp;', leftMessage = '&nbsp;', centerMessage = '&nbsp;', rightMessage = RIGHT_MESSAGE, lowerMessage = '&nbsp;'} = config); // html display element // defines transition and reward functions // defines position of states // initial state of player // object mapping from state names to labels // one of 'never', 'hover', 'click', 'always' // subtracted from score every time a state is clicked // object mapping from edge names (s0 + '__' + s1) to labels // one of 'never', 'hover', 'click', 'always' // subtracted from score every time an edge is clicked // mapping from actions to keycodes // number of trial (starts from 1) // determines the size of states, text, etc...
      LOG_INFO('NAME', this.name);
      SIZE = size;
      _.extend(this, config);
      checkObj(this);
      if (this.stateLabels === 'reward') {
        this.stateLabels = this.stateRewards;
      }
      this.stateLabels[0] = '';
      if (this.energyLimit) {
        leftMessage = 'Energy: <b><span id=mouselab-energy/></b>';
        if (this._block.energyLeft == null) {
          this._block.energyLeft = this.energyLimit;
        }
      } else {
        leftMessage = `Round ${this._block.trialCount + 1}/${this._block.timeline.length}`;
      }
      this.data = {
        // stim: @stim._json
        block: blockName,
        trialIndex: this.trialIndex,
        score: 0,
        rewards: [],
        path: [],
        rt: [],
        actions: [],
        actionTimes: [],
        simulationMode: [],
        queries: {
          click: {
            state: {
              'target': [],
              'time': []
            },
            edge: {
              'target': [],
              'time': []
            }
          },
          mouseover: {
            state: {
              'target': [],
              'time': []
            },
            edge: {
              'target': [],
              'time': []
            }
          },
          mouseout: {
            state: {
              'target': [],
              'time': []
            },
            edge: {
              'target': [],
              'time': []
            }
          }
        }
      };
      if ($('#mouselab-msg-right').length) { // right message already exists
        this.leftMessage = $('#mouselab-msg-left');
        this.leftMessage.html(leftMessage);
        this.centerMessage = $('#mouselab-msg-center');
        this.centerMessage.html(centerMessage);
        this.rightMessage = $('#mouselab-msg-right');
        this.rightMessage.html(rightMessage);
        this.stage = $('#mouselab-stage');
        this.prompt = $('#mouselab-prompt');
        this.prompt.html(prompt);
        console.log('prompt', prompt);
      } else {
        // @canvasElement = $('#mouselab-canvas')
        // @lowerMessage = $('#mouselab-msg-bottom')
        this.display.empty();
        // @display.css 'width', '1000px'

        // leftMessage = "Round: #{@trialIndex + 1}/#{@_block.timeline.length}"
        this.leftMessage = $('<div>', {
          id: 'mouselab-msg-left',
          class: 'mouselab-header',
          html: leftMessage
        }).appendTo(this.display);
        this.centerMessage = $('<div>', {
          id: 'mouselab-msg-center',
          class: 'mouselab-header',
          html: centerMessage
        }).appendTo(this.display);
        this.rightMessage = $('<div>', {
          id: 'mouselab-msg-right',
          class: 'mouselab-header',
          html: rightMessage
        }).appendTo(this.display);
        this.prompt = $('<div>', {
          id: 'mouselab-prompt',
          class: 'mouselab-prompt',
          html: prompt
        }).appendTo(this.display);
        this.stage = $('<div>', {
          id: 'mouselab-stage'
        }).appendTo(this.display);
        if (this.timeLimit) {
          TIME_LEFT = this.timeLimit;
        }
      }
      this.canvasElement = $('<canvas>', {
        id: 'mouselab-canvas'
      }).attr({
        width: 500,
        height: 500
      }).appendTo(this.stage);
      this.lowerMessage = $('<div>', {
        id: 'mouselab-msg-bottom',
        html: lowerMessage || '&nbsp'
      }).appendTo(this.stage);
      this.defaultLowerMessage = lowerMessage;
      mdp = this;
      LOG_INFO('new MouselabMDP', this);
      this.invKeys = _.invert(this.keys);
      this.resetScore();
      this.spendEnergy(0);
      this.freeze = false;
      this.lowerMessage.css('color', '#000');
    }

    startTimer() {
      var $timer, formatTime, start, tick;
      $timer = $('#mouselab-msg-left');
      start = getTime();
      this.timerID = null;
      formatTime = function(time) {
        var minutes, seconds;
        seconds = time % 60;
        minutes = Math.floor(time / 60);
        return (minutes ? (minutes > 9 ? minutes : '0' + minutes) : '00') + ':' + (seconds > 9 ? seconds : '0' + seconds);
      };
      tick = () => {
        if (TIME_LEFT < 0) {
          console.log('DONE');
          window.clearInterval(this.timerID);
          this.lowerMessage.html("<b>Time is up! Press any key to continue.</br>");
          return this.endBlock();
        } else {
          $timer.html(`Time: <b>${formatTime(TIME_LEFT)}</b>`);
          return TIME_LEFT -= 1;
        }
      };
      tick();
      return this.timerID = window.setInterval(tick, 1000);
    }

    endBlock() {
      console.log('endBlock');
      this.blockOver = true;
      jsPsych.pluginAPI.cancelAllKeyboardResponses();
      return this.keyListener = jsPsych.pluginAPI.getKeyboardResponse({
        valid_responses: ['space'],
        rt_method: 'date',
        persist: false,
        allow_held_key: false,
        callback_function: (info) => {
          console.log('CLEAR');
          jsPsych.finishTrial(this.data);
          this.display.empty();
          return jsPsych.endCurrentTimeline();
        }
      });
    }

    handleKey(s0, a) {
      var _, s1;
      LOG_INFO('handleKey', s0, a);
      if (a === 'simulate') {
        if (this.simulationMode) {
          return this.endSimulationMode();
        } else {
          return this.startSimulationMode();
        }
      } else {
        if (!this.simulationMode) {
          this.allowSimulation = false;
          if (this.defaultLowerMessage) {
            this.lowerMessage.html('<b>Move with the arrow keys.</b>');
            this.lowerMessage.css('color', '#000');
          }
        }
        this.data.actions.push(a);
        this.data.simulationMode.push(this.simulationMode);
        this.data.actionTimes.push(Date.now() - this.initTime);
        [_, s1] = this.graph[s0][a];
        // LOG_DEBUG "#{s0}, #{a} -> #{r}, #{s1}"
        return this.move(s0, a, s1);
      }
    }

    startSimulationMode() {
      this.simulationMode = true;
      this.player.set('top', this.states[this.initial].top - 20).set('left', this.states[this.initial].left);
      this.player.set('opacity', 0.4);
      this.canvas.renderAll();
      this.arrive(this.initial);
      // @centerMessage.html 'Ghost Score: <span id=mouselab-ghost-score/>'
      this.rightMessage.html('Ghost Score: <span id=mouselab-score/>');
      this.resetScore();
      this.drawScore(this.data.score);
      return this.lowerMessage.html("<b>👻 Ghost Mode 👻</b>\n<br>\nPress <code>space</code> to return to your corporeal form.");
    }

    endSimulationMode() {
      this.simulationMode = false;
      this.player.set('top', this.states[this.initial].top).set('left', this.states[this.initial].left);
      this.player.set('opacity', 1);
      this.canvas.renderAll();
      this.arrive(this.initial);
      this.centerMessage.html('');
      this.rightMessage.html(RIGHT_MESSAGE);
      this.resetScore();
      return this.lowerMessage.html(this.defaultLowerMessage);
    }

    getOutcome(s0, a) {
      var r, s1;
      LOG_DEBUG(`getOutcome ${s0}, ${a}`);
      [s1, r] = this.graph[s0][a];
      if (this.stateRewards != null) {
        console.log('YES');
        r = this.stateRewards[s1];
      }
      return [r, s1];
    }

    getReward(s0, a, s1) {
      if (this.stateRewards != null) {
        return this.stateRewards[s1];
      } else {
        return this.graph[s0][a];
      }
    }

    move(s0, a, s1) {
      var nClick, newTop, notEnoughClicks, r, s1g;
      this.moved = true;
      if (this.freeze) {
        LOG_INFO('freeze!');
        this.arrive(s0, 'repeat');
        return;
      }
      nClick = this.data.queries.click.state.target.length;
      notEnoughClicks = (this.special.startsWith('trainClick')) && nClick < 3;
      if (notEnoughClicks) {
        this.lowerMessage.html('<b>Inspect at least three nodes before moving!</b>');
        this.lowerMessage.css('color', '#FC4754');
        this.special = 'trainClickBlock';
        this.arrive(s0, 'repeat');
        return;
      }
      r = this.getReward(s0, a, s1);
      LOG_DEBUG(`move ${s0}, ${s1}, ${r}`);
      s1g = this.states[s1];
      this.freeze = true;
      newTop = this.simulationMode ? s1g.top - 20 : s1g.top;
      return this.player.animate({
        left: s1g.left,
        top: newTop
      }, {
        duration: this.moveDelay,
        onChange: this.canvas.renderAll.bind(this.canvas),
        onComplete: () => {
          this.data.rewards.push(r);
          this.addScore(r);
          this.spendEnergy(this.moveEnergy);
          return this.arrive(s1);
        }
      });
    }

    clickState(g, s) {
      var r;
      LOG_INFO(`clickState ${s}`);
      if (this.moved) {
        this.lowerMessage.html("<b>You can't use the node inspector after moving!</b>");
        this.lowerMessage.css('color', '#FC4754');
        return;
      }
      if (this.complete || (`${s}` === `${this.initial}`) || this.freeze) {
        return;
      }
      if (this.special === 'trainClickBlock' && this.data.queries.click.state.target.length === 2) {
        this.lowerMessage.html('<b>Nice job! You can click on more nodes or start moving.</b>');
        this.lowerMessage.css('color', '#000');
      }
      if (this.stateLabels && this.stateDisplay === 'click' && !g.label.text) {
        this.addScore(-this.stateClickCost);
        this.recordQuery('click', 'state', s);
        this.spendEnergy(this.clickEnergy);
        r = this.stateLabels[s];
        if (this.clickDelay) {
          this.freeze = true;
          g.setLabel('...');
          return delay(this.clickDelay, () => {
            this.freeze = false;
            g.setLabel(r);
            return this.canvas.renderAll();
          });
        } else {
          return g.setLabel(r);
        }
      }
    }

    mouseoverState(g, s) {
      // LOG_DEBUG "mouseoverState #{s}"
      if (this.stateLabels && this.stateDisplay === 'hover') {
        // webppl.run('flip()', (s, x) -> g.setLabel (Number x))
        g.setLabel(this.stateLabels[s]);
        return this.recordQuery('mouseover', 'state', s);
      }
    }

    mouseoutState(g, s) {
      // LOG_DEBUG "mouseoutState #{s}"
      if (this.stateLabels && this.stateDisplay === 'hover') {
        g.setLabel('');
        return this.recordQuery('mouseout', 'state', s);
      }
    }

    clickEdge(g, s0, r, s1) {
      var ref;
      if (!this.complete && g.label.text === '?') {
        LOG_DEBUG(`clickEdge ${s0} ${r} ${s1}`);
        if (this.edgeLabels && this.edgeDisplay === 'click' && ((ref = g.label.text) === '?' || ref === '')) {
          g.setLabel(this.getEdgeLabel(s0, r, s1));
          return this.recordQuery('click', 'edge', `${s0}__${s1}`);
        }
      }
    }

    mouseoverEdge(g, s0, r, s1) {
      // LOG_DEBUG "mouseoverEdge #{s0} #{r} #{s1}"
      if (this.edgeLabels && this.edgeDisplay === 'hover') {
        g.setLabel(this.getEdgeLabel(s0, r, s1));
        return this.recordQuery('mouseover', 'edge', `${s0}__${s1}`);
      }
    }

    mouseoutEdge(g, s0, r, s1) {
      // LOG_DEBUG "mouseoutEdge #{s0} #{r} #{s1}"
      if (this.edgeLabels && this.edgeDisplay === 'hover') {
        g.setLabel('');
        return this.recordQuery('mouseout', 'edge', `${s0}__${s1}`);
      }
    }

    getEdgeLabel(s0, r, s1) {
      if (this.edgeLabels === 'reward') {
        return '®';
      } else {
        return this.edgeLabels[`${s0}__${s1}`];
      }
    }

    recordQuery(queryType, targetType, target) {
      this.canvas.renderAll();
      // LOG_DEBUG "recordQuery #{queryType} #{targetType} #{target}"
      // @data["#{queryType}_#{targetType}_#{target}"]
      this.data.queries[queryType][targetType].target.push(target);
      return this.data.queries[queryType][targetType].time.push(Date.now() - this.initTime);
    }

    arrive(s, repeat = false) {
      var a, g, keys;
      g = this.states[s];
      console.log('g', g);
      g.setLabel(this.stateRewards[s]);
      this.canvas.renderAll();
      this.freeze = false;
      LOG_DEBUG('arrive', s);
      if (!repeat) { // sending back to previous state
        this.data.path.push(s);
      }
      // Get available actions.
      if (this.graph[s]) {
        keys = (function() {
          var i, len, ref, results;
          ref = Object.keys(this.graph[s]);
          results = [];
          for (i = 0, len = ref.length; i < len; i++) {
            a = ref[i];
            results.push(this.keys[a]);
          }
          return results;
        }).call(this);
      } else {
        keys = [];
      }
      if (this.allowSimulation) {
        keys.push('space');
      }
      if (!keys.length) {
        this.complete = true;
        this.checkFinished();
        return;
      }
      // Start key listener.
      return this.keyListener = jsPsych.pluginAPI.getKeyboardResponse({
        valid_responses: keys,
        rt_method: 'date',
        persist: false,
        allow_held_key: false,
        callback_function: (info) => {
          var action;
          action = this.invKeys[info.key];
          LOG_DEBUG('key', info.key);
          this.data.rt.push(info.rt);
          return this.handleKey(s, action);
        }
      });
    }

    addScore(v) {
      var score;
      console.log('addScore', v, SCORE);
      this.data.score += v;
      if (this.simulationMode) {
        score = this.data.score;
      } else {
        SCORE += v;
        score = SCORE;
      }
      return this.drawScore(score);
    }

    resetScore() {
      this.data.score = 0;
      return this.drawScore(SCORE);
    }

    drawScore(score) {
      $('#mouselab-score').html('$' + score);
      return $('#mouselab-score').css('color', redGreen(score));
    }

    spendEnergy(v) {
      this._block.energyLeft -= v;
      if (this._block.energyLeft <= 0) {
        LOG_INFO('OUT OF ENERGY');
        this._block.energyLeft = 0;
        this.freeze = true;
        this.lowerMessage.html("<b>You're out of energy! Press</b> <code>space</code> <b>to continue.</br>");
        this.endBlock();
      }
      return $('#mouselab-energy').html(this._block.energyLeft);
    }

    run() {
      // document.body.style.cursor = 'crosshair'
      jsPsych.pluginAPI.cancelAllKeyboardResponses();
      LOG_DEBUG('run');
      this.buildMap();
      fabric.Image.fromURL(this.playerImage, ((img) => {
        this.initPlayer(img);
        this.canvas.renderAll();
        this.initTime = Date.now();
        return this.arrive(this.initial);
      }));
      if (this.timeLimit) {
        return this.startTimer();
      }
    }

    draw(obj) {
      this.canvas.add(obj);
      return obj;
    }

    initPlayer(img) {
      var left, top;
      LOG_DEBUG('initPlayer');
      top = this.states[this.initial].top;
      left = this.states[this.initial].left;
      img.scale(0.3);
      img.set('top', top).set('left', left);
      this.draw(img);
      return this.player = img;
    }

    buildMap() {
      var a, actions, height, location, r, ref, ref1, results, s, s0, s1, width, x, xs, y, ys;
      // Resize canvas.
      [xs, ys] = _.unzip(_.values(this.layout));
      [width, height] = [(_.max(xs)) + 1, (_.max(ys)) + 1];
      this.canvasElement.attr({
        width: width * SIZE,
        height: height * SIZE
      });
      this.canvas = new fabric.Canvas('mouselab-canvas', {
        selection: false
      });
      this.canvas.defaultCursor = 'pointer';
      this.states = {};
      ref = removePrivate(this.layout);
      for (s in ref) {
        location = ref[s];
        [x, y] = location;
        this.states[s] = new State(s, x, y, {
          fill: '#bbb',
          label: this.stateDisplay === 'always' ? this.stateLabels[s] : ''
        });
      }
      ref1 = removePrivate(this.graph);
      results = [];
      for (s0 in ref1) {
        actions = ref1[s0];
        results.push((function() {
          var results1;
          results1 = [];
          for (a in actions) {
            [r, s1] = actions[a];
            results1.push(new Edge(this.states[s0], r, this.states[s1], {
              label: this.edgeDisplay === 'always' ? this.getEdgeLabel(s0, r, s1) : ''
            }));
          }
          return results1;
        }).call(this));
      }
      return results;
    }

    endTrial() {
      window.clearInterval(this.timerID);
      if (this.blockOver) {
        return;
      }
      this.lowerMessage.html("You made <span class=mouselab-score/> on this round.\n<br>\n<b>Press</b> <code>space</code> <b>to continue.</b>");
      $('.mouselab-score').html('$' + this.data.score);
      $('.mouselab-score').css('color', redGreen(this.data.score));
      $('.mouselab-score').css('font-weight', 'bold');
      return this.keyListener = jsPsych.pluginAPI.getKeyboardResponse({
        valid_responses: ['space'],
        rt_method: 'date',
        persist: false,
        allow_held_key: false,
        callback_function: (info) => {
          console.log('CLEAR');
          jsPsych.finishTrial(this.data);
          return this.stage.empty();
        }
      });
    }

    checkFinished() {
      if (this.complete) {
        return this.endTrial();
      }
    }

  };
  //  =========================== #
  //  ========= Graphics ========= #
  //  =========================== #
  State = class State {
    constructor(name, left, top, config = {}) {
      var conf;
      this.name = name;
      left = (left + 0.5) * SIZE;
      top = (top + 0.5) * SIZE;
      conf = {
        left: left,
        top: top,
        fill: '#bbbbbb',
        radius: SIZE / 4,
        label: ''
      };
      _.extend(conf, config);
      // Due to a quirk in Fabric, the maximum width of the label
      // is set when the object is initialized (the call to super).
      // Thus, we must initialize the label with a placeholder, then
      // set it to the proper value afterwards.
      this.circle = new fabric.Circle(conf);
      this.label = new Text('----------', left, top, {
        fontSize: SIZE / 4,
        fill: '#44d'
      });
      this.radius = this.circle.radius;
      this.left = this.circle.left;
      this.top = this.circle.top;
      mdp.canvas.add(this.circle);
      mdp.canvas.add(this.label);
      this.circle.on('mousedown', () => {
        return mdp.clickState(this, this.name);
      });
      this.circle.on('mouseover', () => {
        return mdp.mouseoverState(this, this.name);
      });
      this.circle.on('mouseout', () => {
        return mdp.mouseoutState(this, this.name);
      });
      this.setLabel(conf.label);
    }

    setLabel(txt, conf = {}) {
      var post, pre;
      LOG_DEBUG('setLabel', txt);
      ({pre = '', post = ''} = conf);
      // LOG_DEBUG "setLabel #{txt}"
      if (txt != null) {
        this.label.setText(`${pre}${txt}${post}`);
        this.label.setFill(redGreen(txt));
      } else {
        this.label.setText('');
      }
      return this.dirty = true;
    }

  };
  Edge = class Edge {
    constructor(c1, reward, c2, config = {}) {
      var adjX, adjY, ang, labX, labY, label, rotateLabel, spacing, x1, x2, y1, y2;
      ({spacing = 8, adjX = 0, adjY = 0, rotateLabel = false, label = ''} = config);
      [x1, y1, x2, y2] = [c1.left + adjX, c1.top + adjY, c2.left + adjX, c2.top + adjY];
      this.arrow = new Arrow(x1, y1, x2, y2, c1.radius + spacing, c2.radius + spacing);
      ang = (this.arrow.ang + Math.PI / 2) % (Math.PI * 2);
      if ((0.5 * Math.PI <= ang && ang <= 1.5 * Math.PI)) {
        ang += Math.PI;
      }
      [labX, labY] = polarMove(x1, y1, angle(x1, y1, x2, y2), SIZE * 0.45);
      // See note about placeholder in State.
      this.label = new Text('----------', labX, labY, {
        angle: rotateLabel ? ang * 180 / Math.PI : 0,
        fill: redGreen(label),
        fontSize: SIZE / 4,
        textBackgroundColor: 'white'
      });
      this.arrow.on('mousedown', () => {
        return mdp.clickEdge(this, c1.name, reward, c2.name);
      });
      this.arrow.on('mouseover', () => {
        return mdp.mouseoverEdge(this, c1.name, reward, c2.name);
      });
      this.arrow.on('mouseout', () => {
        return mdp.mouseoutEdge(this, c1.name, reward, c2.name);
      });
      this.setLabel(label);
      mdp.canvas.add(this.arrow);
      mdp.canvas.add(this.label);
    }

    setLabel(txt, conf = {}) {
      var post, pre;
      ({pre = '', post = ''} = conf);
      // LOG_DEBUG "setLabel #{txt}"
      if (txt) {
        this.label.setText(`${pre}${txt}${post}`);
        this.label.setFill(redGreen(txt));
      } else {
        this.label.setText('');
      }
      return this.dirty = true;
    }

  };
  Arrow = class Arrow extends fabric.Group {
    constructor(x1, y1, x2, y2, adj1 = 0, adj2 = 0) {
      var ang, centerX, centerY, deltaX, deltaY, dx, dy, line, point;
      ang = angle(x1, y1, x2, y2);
      [x1, y1] = polarMove(x1, y1, ang, adj1);
      [x2, y2] = polarMove(x2, y2, ang, -(adj2 + 7.5));
      line = new fabric.Line([x1, y1, x2, y2], {
        stroke: '#555',
        selectable: false,
        strokeWidth: 3
      });
      centerX = (x1 + x2) / 2;
      centerY = (y1 + y2) / 2;
      deltaX = line.left - centerX;
      deltaY = line.top - centerY;
      dx = x2 - x1;
      dy = y2 - y1;
      point = new fabric.Triangle({
        left: x2 + deltaX,
        top: y2 + deltaY,
        pointType: 'arrow_start',
        angle: ang * 180 / Math.PI,
        width: 10,
        height: 10,
        fill: '#555'
      });
      super([line, point]);
      this.ang = ang;
      this.centerX = centerX;
      this.centerY = centerY;
    }

  };
  Text = class Text extends fabric.Text {
    constructor(txt, left, top, config) {
      var conf;
      txt = String(txt);
      conf = {
        left: left,
        top: top,
        fontFamily: 'helvetica',
        fontSize: SIZE / 8
      };
      _.extend(conf, config);
      super(txt, conf);
    }

  };
  // ================================= #
  // ========= jsPsych stuff ========= #
  // ================================= #
  plugin = {
    trial: function(display_element, trialConfig) {
      var trial;
      // trialConfig = jsPsych.pluginAPI.evaluateFunctionParameters trialConfig, ['_init', 'constructor']
      trialConfig.display = display_element;
      LOG_INFO('trialConfig', trialConfig);
      trial = new MouselabMDP(trialConfig);
      trial.run();
      if (trialConfig._block) {
        trialConfig._block.trialCount += 1;
      }
      return TRIAL_INDEX += 1;
    }
  };
  return plugin;
})();

// ---
// generated by js2coffee 2.2.0