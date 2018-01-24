// Generated by CoffeeScript 1.12.3
var BLOCKS, CONDITION, DEBUG, DEMO, N_TRIAL, PARAMS, SCORE, STRUCTURE, TRIALS, calculateBonus, createStartButton, getTrials, initializeExperiment, psiturk, saveData,
  extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

DEBUG = false;

if (DEBUG) {
  console.log("X X X X X X X X X X X X X X X X X\n X X X X X DEBUG  MODE X X X X X\nX X X X X X X X X X X X X X X X X");
  CONDITION = 0;
} else {
  console.log("# =============================== #\n# ========= NORMAL MODE ========= #\n# =============================== #");
  console.log('16/01/18 12:38:03 PM');
  CONDITION = parseInt(condition);
}

if (mode === "{{ mode }}") {
  DEMO = true;
  CONDITION = 0;
}

BLOCKS = void 0;

PARAMS = void 0;

TRIALS = void 0;

STRUCTURE = void 0;

N_TRIAL = void 0;

SCORE = 0;

calculateBonus = void 0;

getTrials = void 0;

psiturk = new PsiTurk(uniqueId, adServerLoc, mode);

saveData = function() {
  return new Promise(function(resolve, reject) {
    var timeout;
    timeout = delay(10000, function() {
      return reject('timeout');
    });
    return psiturk.saveData({
      error: function() {
        clearTimeout(timeout);
        console.log('Error saving data!');
        return reject('error');
      },
      success: function() {
        clearTimeout(timeout);
        console.log('Data saved to psiturk server.');
        return resolve();
      }
    });
  });
};

$(window).resize(function() {
  return checkWindowSize(800, 700, $('#jspsych-target'));
});

$(window).resize();

$(window).on('load', function() {
  var loadTimeout, slowLoad;
  slowLoad = function() {
    var ref;
    return (ref = $('slow-load')) != null ? ref.show() : void 0;
  };
  loadTimeout = delay(12000, slowLoad);
  psiturk.preloadImages(['static/images/spider.png']);
  return delay(300, function() {
    console.log('Loading data');
    PARAMS = {
      inspectCost: 1,
      startTime: Date(Date.now()),
      bonusRate: .001,
      variance: ['constant_high', 'constant_low', 'increasing', 'decreasing'][CONDITION]
    };
    psiturk.recordUnstructuredData('params', PARAMS);
    STRUCTURE = loadJson("static/json/binary_structure.json");
    TRIALS = loadJson("static/json/binary_trees_" + PARAMS.variance + ".json");
    console.log("loaded " + (TRIALS != null ? TRIALS.length : void 0) + " trials");
    getTrials = (function() {
      var idx, t;
      t = _.shuffle(TRIALS);
      idx = 0;
      return function(n) {
        idx += n;
        return t.slice(idx - n, idx);
      };
    })();
    if (DEBUG) {
      createStartButton();
      return clearTimeout(loadTimeout);
    } else {
      console.log('Testing saveData');
      if (DEMO) {
        clearTimeout(loadTimeout);
        return delay(500, createStartButton);
      } else {
        return saveData().then(function() {
          clearTimeout(loadTimeout);
          return delay(500, createStartButton);
        })["catch"](function() {
          clearTimeout(loadTimeout);
          return $('#data-error').show();
        });
      }
    }
  });
});

createStartButton = function() {
  if (DEBUG) {
    initializeExperiment();
    return;
  }
  $('#load-icon').hide();
  $('#slow-load').hide();
  $('#success-load').show();
  return $('#load-btn').click(initializeExperiment);
};

initializeExperiment = function() {
  var Block, ButtonBlock, MouselabBlock, QuizLoop, TextBlock, bonus_text, divider, experiment_timeline, finish, fullMessage, img, prompt_resubmit, quiz, reprompt, reset_score, save_data, test, text, train, train_basic, train_final, train_ghost, train_hidden, train_inspect_cost, train_inspector, verbal_responses;
  $('#jspsych-target').html('');
  console.log('INITIALIZE EXPERIMENT');
  text = {
    debug: function() {
      if (DEBUG) {
        return "`DEBUG`";
      } else {
        return '';
      }
    }
  };
  Block = (function() {
    function Block(config) {
      _.extend(this, config);
      this._block = this;
      if (this._init != null) {
        this._init();
      }
    }

    return Block;

  })();
  TextBlock = (function(superClass) {
    extend(TextBlock, superClass);

    function TextBlock() {
      return TextBlock.__super__.constructor.apply(this, arguments);
    }

    TextBlock.prototype.type = 'text';

    TextBlock.prototype.cont_key = [];

    return TextBlock;

  })(Block);
  ButtonBlock = (function(superClass) {
    extend(ButtonBlock, superClass);

    function ButtonBlock() {
      return ButtonBlock.__super__.constructor.apply(this, arguments);
    }

    ButtonBlock.prototype.type = 'button-response';

    ButtonBlock.prototype.is_html = true;

    ButtonBlock.prototype.choices = ['Continue'];

    ButtonBlock.prototype.button_html = '<button class="btn btn-primary btn-lg">%choice%</button>';

    return ButtonBlock;

  })(Block);
  QuizLoop = (function(superClass) {
    extend(QuizLoop, superClass);

    function QuizLoop() {
      return QuizLoop.__super__.constructor.apply(this, arguments);
    }

    QuizLoop.prototype.loop_function = function(data) {
      var c, i, len, ref;
      console.log('data', data);
      ref = data[data.length].correct;
      for (i = 0, len = ref.length; i < len; i++) {
        c = ref[i];
        if (!c) {
          return true;
        }
      }
      return false;
    };

    return QuizLoop;

  })(Block);
  MouselabBlock = (function(superClass) {
    extend(MouselabBlock, superClass);

    function MouselabBlock() {
      return MouselabBlock.__super__.constructor.apply(this, arguments);
    }

    MouselabBlock.prototype.type = 'mouselab-mdp';

    MouselabBlock.prototype.playerImage = 'static/images/spider.png';

    MouselabBlock.prototype.lowerMessage = "Click on the nodes to reveal their values.<br>\nMove with the arrow keys.";

    MouselabBlock.prototype._init = function() {
      _.extend(this, STRUCTURE);
      return this.trialCount = 0;
    };

    return MouselabBlock;

  })(Block);
  img = function(name) {
    return "<img class='display' src='static/images/" + name + ".png'/>";
  };
  fullMessage = "";
  reset_score = new Block({
    type: 'call-function',
    func: function() {
      return SCORE = 0;
    }
  });
  divider = new TextBlock({
    text: function() {
      SCORE = 0;
      return "<div class='center'>Press <code>space</code> to continue.</div>";
    }
  });
  train_basic = new MouselabBlock({
    blockName: 'train_basic',
    stateDisplay: 'always',
    prompt: function() {
      return markdown("## Web of Cash\n\nIn this HIT, you will play a game called *Web of Cash*. You will guide a\nmoney-loving spider through a spider web. When you land on a gray circle\n(a ***node***) the value of the node is added to your score. You can\nmove the spider with the arrow keys, but only in the direction of the\narrows between the nodes. Go ahead, try a few rounds now!");
    },
    lowerMessage: '<b>Move with the arrow keys.</b>',
    timeline: getTrials(10)
  });
  train_hidden = new MouselabBlock({
    blockName: 'train_hidden',
    allowSimulation: false,
    stateDisplay: 'never',
    prompt: function() {
      return markdown("## Hidden Information\n\nNice job! When you can see the values of each node, it's not too hard to\ntake the best possible path. Unfortunately, you can't always see the\nvalue of the nodes. Without this information, it's hard to make good\ndecisions. Try completing a few more rounds.");
    },
    lowerMessage: '<b>Move with the arrow keys.</b>',
    timeline: getTrials(5)
  });
  train_ghost = new MouselabBlock({
    blockName: 'train_ghost',
    stateDisplay: 'never',
    prompt: function() {
      return markdown("## Ghost Mode\n\nIt's hard to make good decisions when you can't see what you're doing!\nFortunately, you have been equipped with a very handy tool. By pressing\n`space` you will enter ***Ghost Mode***. While in Ghost Mode your true\nscore won't change, but you'll see how your score *would have* changed\nif you had visited that node for real. At any point you can press\n`space` again to return to the realm of the living. **Note:** You can\nonly enter Ghost Mode when you are in the first node.");
    },
    lowerMessage: '<b>Press</b> <code>space</code>  <b>to enter ghost mode.</b>',
    timeline: getTrials(5)
  });
  train_inspector = new MouselabBlock({
    blockName: 'train_inspector',
    special: 'trainClick',
    stateDisplay: 'click',
    stateClickCost: 0,
    prompt: function() {
      return markdown("## Node Inspector\n\nIt's hard to make good decision when you can't see what you're doing!\nFortunately, you have access to a ***node inspector*** which can reveal\nthe value of a node. To use the node inspector, simply click on a node.\n**Note:** you can only use the node inspector when you're on the first\nnode.\n\nPractice using the inspector on **at least three nodes** before moving.");
    },
    timeline: getTrials(5)
  });
  train_inspect_cost = new MouselabBlock({
    blockName: 'train_inspect_cost',
    stateDisplay: 'click',
    stateClickCost: PARAMS.inspectCost,
    prompt: function() {
      return markdown("## The price of information\n\nYou can use node inspector to gain information and make better\ndecisions. But, as always, there's a catch. Using the node inspector\ncosts $" + PARAMS.inspectCost + " per node. To maximize your score, you have\nto know when it's best to gather more information, and when it's time to\nact!");
    },
    timeline: getTrials(5)
  });
  bonus_text = function(long) {
    var s;
    if (PARAMS.bonusRate !== .001) {
      throw new Error('Incorrect bonus rate');
    }
    s = "**you will earn 1 cent for every $10 you make in the game.**";
    if (long) {
      s += " For example, if your final score is $500, you will receive a bonus of **$0.50**.";
    }
    return s;
  };
  train_final = new MouselabBlock({
    blockName: 'train_final',
    stateDisplay: 'click',
    stateClickCost: PARAMS.inspectCost,
    prompt: function() {
      return markdown("## Earn a Big Bonus\n\nNice! You've learned how to play *Web of Cash*, and you're almost ready\nto play it for real. To make things more interesting, you will earn real\nmoney based on how well you play the game. Specifically,\n" + (bonus_text('long')) + "\n\nThis is the **final practice round** before your score starts counting\ntowards your bonus.");
    },
    lowerMessage: fullMessage,
    timeline: getTrials(5)
  });
  train = new Block({
    training: true,
    timeline: [
      train_basic, divider, train_hidden, divider, train_inspector, divider, train_inspect_cost, divider, train_final, new ButtonBlock({
        stimulus: function() {
          SCORE = 0;
          psiturk.finishInstructions();
          return markdown("# Training Completed\n\nWell done! You've completed the training phase and you're ready to\nplay *Web of Cash* for real. You will have **" + test.timeline.length + "\nrounds** to make as much money as you can. Remember, " + (bonus_text()) + "\nGood luck!");
        }
      })
    ]
  });
  quiz = new Block({
    preamble: function() {
      return markdown("# Quiz");
    },
    type: 'survey-multi-choice',
    questions: ["What was the range of node values?", "What is the cost of clicking?", "How much REAL money do you earn?"],
    options: [['$0 to $10', '-$5 to $5', '-$12 to 12', '-$30 to $30'], ['$0', '$1', '$2', '$3'], ['1 cent for every $100 you make in the game', '1 cent for every $10 you make in the game', '1 dollar for every $10 you make in the game']]
  });
  test = new MouselabBlock({
    blockName: 'test',
    stateDisplay: 'click',
    stateClickCost: PARAMS.inspectCost,
    timeline: getTrials(20)
  });
  verbal_responses = new Block({
    type: 'survey-text',
    preamble: function() {
      return markdown("# Please answer these questions\n");
    },
    questions: ['How did you decide when to stop clicking?', 'How did you decide where to click?', 'How did you decide where NOT to click?', 'Where were you most likely to click at the beginning of each trial?', 'Can you describe your strategy?'],
    button: 'Finish'
  });
  finish = new Block({
    type: 'survey-text',
    preamble: function() {
      return markdown("# You've completed the HIT\n\nThanks for participating. We hope you had fun! Based on your\nperformance, you will be awarded a bonus of\n**$" + (calculateBonus().toFixed(2)) + "**.\n\nPlease briefly answer the questions below before you submit the HIT.");
    },
    questions: ['Was anything confusing or hard to understand?', 'What is your age?', 'Additional coments?'],
    button: 'Submit HIT'
  });
  if (DEBUG) {
    experiment_timeline = [quiz, verbal_responses, finish];
  } else {
    experiment_timeline = [train, quiz, test, verbal_responses, finish];
  }
  calculateBonus = function() {
    var bonus;
    bonus = SCORE * PARAMS.bonusRate;
    bonus = (Math.round(bonus * 100)) / 100;
    return bonus;
  };
  reprompt = null;
  save_data = function() {
    return psiturk.saveData({
      success: function() {
        console.log('Data saved to psiturk server.');
        if (reprompt != null) {
          window.clearInterval(reprompt);
        }
        return psiturk.computeBonus('compute_bonus', psiturk.completeHIT);
      },
      error: function() {
        return prompt_resubmit;
      }
    });
  };
  prompt_resubmit = function() {
    $('#jspsych-target').html("<h1>Oops!</h1>\n<p>\nSomething went wrong submitting your HIT.\nThis might happen if you lose your internet connection.\nPress the button to resubmit.\n</p>\n<button id=\"resubmit\">Resubmit</button>");
    return $('#resubmit').click(function() {
      $('#jspsych-target').html('Trying to resubmit...');
      reprompt = window.setTimeout(prompt_resubmit, 10000);
      return save_data();
    });
  };
  return jsPsych.init({
    display_element: $('#jspsych-target'),
    timeline: experiment_timeline,
    on_finish: function() {
      if (DEBUG) {
        return jsPsych.data.displayData();
      } else {
        psiturk.recordUnstructuredData('final_bonus', calculateBonus());
        return save_data();
      }
    },
    on_data_update: function(data) {
      console.log('data', data);
      return psiturk.recordTrialData(data);
    }
  });
};
