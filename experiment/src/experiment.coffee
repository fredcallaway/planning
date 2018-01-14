###
experiment.coffee
Fred Callaway

Demonstrates the jsych-mdp plugin

###
# coffeelint: disable=max_line_length, indentation


DEBUG = yes
if DEBUG
  console.log """
  X X X X X X X X X X X X X X X X X
   X X X X X DEBUG  MODE X X X X X
  X X X X X X X X X X X X X X X X X
  """
  condition = 0
  counterbalance = 0
else
  console.log """
  # =============================== #
  # ========= NORMAL MODE ========= #
  # =============================== #
  """

if mode is "{{ mode }}"
  DEMO = true
  condition = 1
  counterbalance = 0


# Globals.
psiturk = new PsiTurk uniqueId, adServerLoc, mode

BLOCKS = undefined
PARAMS = undefined
TRIALS = undefined
N_TRIAL = undefined
calculateBonus = undefined
SCORE = 0

# because the order of arguments of setTimeout is awful.
delay = (time, func) -> setTimeout func, time

# $(window).resize -> checkWindowSize 920, 720, $('#jspsych-target')
# $(window).resize()

# $(document).ready ->
$(window).on 'load', ->
  # Load data and test connection to server.
  slowLoad = -> document.getElementById("failLoad").style.display = "block"
  loadTimeout = delay 12000, slowLoad

  psiturk.preloadImages [
    'static/images/example1.png'
    'static/images/example2.png'
    'static/images/example3.png'
    'static/images/money.png'
    'static/images/plane.png'
    'static/images/spider.png'
  ]


  delay 300, ->
    console.log "---------------------------"
    console.log 'Loading data'
    console.log "static/json/condition_#{condition}_#{counterbalance}.json"
    expData = loadJson "static/json/condition_#{condition}_#{counterbalance}.json"
    console.log 'expData', expData
    PARAMS = expData.params

    PARAMS.start_time = Date(Date.now())
    
    BLOCKS = expData.blocks
    psiturk.recordUnstructuredData 'params', PARAMS

    if DEBUG or DEMO
      createStartButton()
      # PARAMS.message = true
    else
      console.log 'Testing saveData'
      ERROR = null
      psiturk.saveData
        error: ->
          console.log 'ERROR saving data.'
          ERROR = true
        success: ->
          console.log 'Data saved to psiturk server.'
          clearTimeout loadTimeout
          delay 500, createStartButton


createStartButton = ->
  if DEBUG
    initializeExperiment()
    return
  document.getElementById("loader").style.display = "none"
  document.getElementById("successLoad").style.display = "block"
  document.getElementById("failLoad").style.display = "none"
  $('#load-btn').click initializeExperiment


initializeExperiment = ->
  $('#jspsych-target').html ''
  console.log 'INITIALIZE EXPERIMENT'

  #  ======================== #
  #  ========= TEXT ========= #
  #  ======================== #

  # These functions will be executed by the jspsych plugin that
  # they are passed to. String interpolation will use the values
  # of global variables defined in this file at the time the function
  # is called.


  text =
    debug: -> if DEBUG then "`DEBUG`" else ''

  # ================================= #
  # ========= BLOCK CLASSES ========= #
  # ================================= #

  class Block
    constructor: (config) ->
      _.extend(this, config)
      @_block = this  # allows trial to access its containing block for tracking state
      if @_init?
        @_init()

  class TextBlock extends Block
    type: 'text'
    cont_key: []

  class ButtonBlock extends Block
    type: 'button-response'
    is_html: true
    choices: ['Continue']
    button_html: '<button class="btn btn-primary btn-lg">%choice%</button>'


  class QuizLoop extends Block
    loop_function: (data) ->
      console.log 'data', data
      for c in data[data.length].correct
        if not c
          return true
      return false

  class MouselabBlock extends Block
    type: 'mouselab-mdp'
    playerImage: 'static/images/spider.png'
    moveDelay: PARAMS.moveDelay
    clickDelay: PARAMS.clickDelay
    moveEnergy: PARAMS.moveEnergy
    clickEnergy: PARAMS.clickEnergy
    _init: -> @trialCount = 0



  #  ============================== #
  #  ========= EXPERIMENT ========= #
  #  ============================== #

  img = (name) -> """<img class='display' src='static/images/#{name}.png'/>"""


  # instruct_loop = new Block
  #   timeline: [instructions, quiz]
  #   loop_function: (data) ->
  #     for c in data[1].correct
  #       if not c
  #         return true  # try again
  #     psiturk.finishInstructions()
  #     psiturk.saveData()
  #     return false

  # fullMessage = """
  #   Click on the nodes to reveal their values.<br>
  #   Move with the arrow keys.
  # """
  fullMessage = ""
  reset_score = new Block
    type: 'call-function'
    func: ->
      SCORE = 0

  divider = new TextBlock
    text: ->
      SCORE = 0
      "<div class='center'>Press <code>space</code> to continue.</div>"

  train_basic = new MouselabBlock
    blockName: 'train_basic'
    allowSimulation: false
    stateDisplay: 'always'
    prompt: ->
      markdown """
      ## Web of Cash

      In this HIT, you will play a game called *Web of Cash*. You will guide
      a money-loving spider through a spider web. When you land on a gray
      circle (a ***node***) the value of the node is added to your score.
      You can move the spider with the arrow keys, but only in the direction
      of the arrows between the nodes. Go ahead, try a few rounds now!
    """
    lowerMessage: '<b>Move with the arrow keys.</b>'
    timeline: BLOCKS.train_basic

  
  train_hidden = new MouselabBlock
    blockName: 'train_hidden'
    allowSimulation: false
    stateDisplay: 'never'
    prompt: ->
      markdown """
      ## Hidden Information

      Nice job! When you can see the values of each node, it's not too hard
      to take the best possible path. Unfortunately, you can't always see
      the value of the nodes. Without this information, it's hard to make
      good decisions. Try completing a few more rounds.
    """
    lowerMessage: '<b>Move with the arrow keys.</b>'
    timeline: BLOCKS.train_hidden

  
  train_ghost = new MouselabBlock
    blockName: 'train_ghost'
    stateDisplay: 'never'
    prompt: ->
      markdown """
      ## Ghost Mode

      It's hard to make good decisions when you can't see what you're
      doing! Fortunately, you have been equipped with a very handy tool.
      By pressing `space` you will enter ***Ghost Mode***. While in Ghost Mode
      your true score won't change, but you'll see how your score *would
      have* changed if you had visited that node for real.
      At any point you can press `space` again to return to the realm of the living.
      **Note:** You can only enter Ghost Mode when you are in the first node.
    """
    lowerMessage: '<b>Press</b> <code>space</code>  <b>to enter ghost mode.</b>'
    timeline: BLOCKS.train_ghost

  
  train_inspector = new MouselabBlock
    blockName: 'train_inspector'
    stateDisplay: 'click'
    special: 'trainClick'
    prompt: ->
      markdown """
      ## Node Inspector

      It's hard to make good decision when you can't see what you're
      doing! Fortunately, you have access to a ***node inspector*** which
      can reveal the value of a node. To use the node inspector, simply
      click on a node. Practice using the inspector on **at least three**
      nodes before moving.
    """
    # but the node inspector takes some time to work and you can only inspect one node at a time.
    timeline: BLOCKS.train_ghost
    lowerMessage: "<b>Click on the nodes to reveal their values.<b>"


  train_inspect_cost = new MouselabBlock
    blockName: 'train_inspect_cost'
    stateDisplay: 'click'
    # energyLimit: 20
    stateClickCost: PARAMS.inspectCost
    prompt: ->
      markdown """
      ## 4. The price of information

      Sweet! You can use node inspector to gain information and make
      better decisions. But, as always, there's a catch. The node inspetor
      costs $#{PARAMS.inspectCost} per node. To maximize your score, you
      have to know when it's best to gather more infromation, and when
      it's time to act!

    """
    lowerMessage: '<b>Play until you run out of energy.</b>'
    timeline: BLOCKS.train_ghost


  bonus_text = (long) ->
    if PARAMS.bonus_rate isnt .001
      throw new Error('Incorrect bonus rate')
    s = "**you will earn 1 cent for every $10 you make in the game.**"
    if long
      s += " For example, if your final score is $700, you will receive a bonus of **$0.70**."
    return s


  train_final = new MouselabBlock
    blockName: 'train_final'
    stateDisplay: 'click'
    # energyLimit: 100
    prompt: ->
      markdown """
      ## Earn a Big Bonus

      Nice! You've learned how to play *Web of Cash*, and you're ready to
      play it for real. To make things more interesting, you will earn
      real money based on how well you play the game. Specifically, 
      #{bonus_text('long')} This is the final
      practice round before your score starts counting towards your bonus.
    """
    lowerMessage: fullMessage
    timeline: BLOCKS.train_final


  train = new Block
    training: true
    timeline: [
      # train_basic
      # divider
      # train_hidden
      # divider
      train_inspector
      train_inspect_cost
      divider
      train_final
      new ButtonBlock
        stimulus: ->
          SCORE = 0
          psiturk.finishInstructions()
          markdown """
          # Training Completed

          Well done! You've completed the training phase and you're ready to
          play *Web of Cash* for real. You will have **#{test.timeline.length} rounds** to make
          as much money as you can.
          Remember, #{bonus_text()} Good luck!
        """
    ]


  test = new MouselabBlock
    blockName: 'test'
    stateDisplay: 'click'
    # energyLimit: 200
    # timeLimit: PARAMS.timeLimit
    lowerMessage: fullMessage
    timeline: if DEBUG then BLOCKS.test.slice(0, 3) else BLOCKS.test


  finish = new Block
    type: 'survey-text'
    preamble: -> markdown """
        # You've completed the HIT

        Thanks for participating. We hope you had fun! Based on your
        performance, you will be awarded a bonus of
        **$#{calculateBonus().toFixed(2)}**.

        Please briefly answer the questions below before you submit the HIT.
      """

    questions: [
      'Was anything confusing or hard to understand?'
      'What was your strategy?'
      'Additional coments?'
    ]
    button: 'Submit HIT'

  # ppl = new Block
  #   type: 'webppl'
  #   file: 'static/model.wppl'

  if DEBUG
    experiment_timeline = [
      train
      test
      finish
    ]
  else
    experiment_timeline = [
      train
      test
      finish
    ]


  # ================================================ #
  # ========= START AND END THE EXPERIMENT ========= #
  # ================================================ #

  # bonus is the total score multiplied by something
  calculateBonus = ->
    bonus = SCORE * PARAMS.bonus_rate
    bonus = (Math.round (bonus * 100)) / 100  # round to nearest cent
    return bonus
  

  reprompt = null
  save_data = ->
    psiturk.saveData
      success: ->
        console.log 'Data saved to psiturk server.'
        if reprompt?
          window.clearInterval reprompt
        psiturk.computeBonus('compute_bonus', psiturk.completeHIT)
      error: -> prompt_resubmit


  prompt_resubmit = ->
    $('#jspsych-target').html """
      <h1>Oops!</h1>
      <p>
      Something went wrong submitting your HIT.
      This might happen if you lose your internet connection.
      Press the button to resubmit.
      </p>
      <button id="resubmit">Resubmit</button>
    """
    $('#resubmit').click ->
      $('#jspsych-target').html 'Trying to resubmit...'
      reprompt = window.setTimeout(prompt_resubmit, 10000)
      save_data()

  jsPsych.init
    display_element: $('#jspsych-target')
    timeline: experiment_timeline
    # show_progress_bar: true

    on_finish: ->
      if DEBUG
        jsPsych.data.displayData()
      else
        psiturk.recordUnstructuredData 'final_bonus', calculateBonus()
        save_data()

    on_data_update: (data) ->
      console.log 'data', data
      psiturk.recordTrialData data

