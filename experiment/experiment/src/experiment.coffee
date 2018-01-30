# coffeelint: disable=max_line_length, indentation

DEBUG = no
TALK = no

if DEBUG
  console.log """
  X X X X X X X X X X X X X X X X X
   X X X X X DEBUG  MODE X X X X X
  X X X X X X X X X X X X X X X X X
  """
  CONDITION = 0

else
  console.log """
  # =============================== #
  # ========= NORMAL MODE ========= #
  # =============================== #
  """
  console.log '16/01/18 12:38:03 PM'
  CONDITION = parseInt condition

if mode is "{{ mode }}"
  DEMO = true
  CONDITION = 0

BLOCKS = undefined
PARAMS = undefined
TRIALS = undefined
STRUCTURE = undefined
N_TRIAL = undefined
SCORE = 0
calculateBonus = undefined
getTrials = undefined

psiturk = new PsiTurk uniqueId, adServerLoc, mode
saveData = ->
  new Promise (resolve, reject) ->
    timeout = delay 10000, ->
      reject('timeout')

    psiturk.saveData
      error: ->
        clearTimeout timeout
        console.log 'Error saving data!'
        reject('error')
      success: ->
        clearTimeout timeout
        console.log 'Data saved to psiturk server.'
        resolve()


$(window).resize -> checkWindowSize 800, 600, $('#jspsych-target')
$(window).resize()
$(window).on 'load', ->
  # Load data and test connection to server.
  slowLoad = -> $('slow-load')?.show()
  loadTimeout = delay 12000, slowLoad

  psiturk.preloadImages [
    'static/images/spider.png'
  ]


  delay 300, ->
    console.log 'Loading data'
        
    PARAMS =
      inspectCost: 1
      startTime: Date(Date.now())
      bonusRate: .002
      variance: ['2_4_24', '24_4_2'][CONDITION]
      branching: '312'

    psiturk.recordUnstructuredData 'params', PARAMS

    id = "#{PARAMS.branching}_#{PARAMS.variance}"
    STRUCTURE = loadJson "static/json/structure/#{id}.json"
    TRIALS = loadJson "static/json/rewards/#{id}.json"
    console.log "loaded #{TRIALS?.length} trials"

    getTrials = do ->
      t = _.shuffle TRIALS
      idx = 0
      return (n) ->
        idx += n
        t.slice(idx-n, idx)

    if DEBUG or TALK
      createStartButton()
      clearTimeout loadTimeout
    else
      console.log 'Testing saveData'
      if DEMO
        clearTimeout loadTimeout
        delay 500, createStartButton
      else
        saveData().then(->
          clearTimeout loadTimeout
          delay 500, createStartButton
        ).catch(->
          clearTimeout loadTimeout
          $('#data-error').show()
        )

createStartButton = ->
  if DEBUG or TALK
    initializeExperiment()
    return
  $('#load-icon').hide()
  $('#slow-load').hide()
  $('#success-load').show()
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
    # moveDelay: PARAMS.moveDelay
    # clickDelay: PARAMS.clickDelay
    # moveEnergy: PARAMS.moveEnergy
    # clickEnergy: PARAMS.clickEnergy
    lowerMessage: """
      Click on the nodes to reveal their values.<br>
      Move with the arrow keys.
    """
    
    _init: ->
      _.extend(this, STRUCTURE)
      @trialCount = 0



  #  ============================== #
  #  ========= EXPERIMENT ========= #
  #  ============================== #

  img = (name) -> """<img class='display' src='static/images/#{name}.png'/>"""
  
  nodeValuesDescription = switch PARAMS.variance
    when "6_6_6" then """
      A node can have value -10, -5, 5, or 10. All values are equally likely.
    """
    when "2_4_24" then """
      The more steps it takes to reach a node, the more variable its value
      tends to be: The value of a node you can reach in **one** step is equally
      likely to be **-4, -2, 2, or 4**. The value of a node you can reach in **two**
      steps is equally likely to be **-8, -4, 4, or 8**. Finally,  the value of a
      node you can reach in **three** steps is equally likely to be **-48, -24, 24,
      or 48**.
    """
    when "24_4_2" then """
      The more steps it takes to reach a node, the less variable its value
      tends to be: The value of a node you can reach in **one** step is equally
      likely to be **-48, -24, 24, or 48**. The value of a node you can reach in
      **two** steps is equally likely to be **-8, -4, 4, or 8**. Finally,  the value
      of a node you can reach in **three** steps is equally likely to be  -4, -2,
      2, or 4.
    """
        
  # instruct_loop = new Block
  #   timeline: [instructions, quiz]
  #   loop_function: (data) ->
  #     for c in data[1].correct
  #       if not c
  #         return true  # try again
  #     psiturk.finishInstructions()
  #     psiturk.saveData()
  #     return false

  fullMessage = ""
  reset_score = new Block
    type: 'call-function'
    func: ->
      SCORE = 0

  divider = new TextBlock
    text: ->
      SCORE = 0
      "<div class='center'>Press <code>space</code> to continue.</div>"

  train_basic1 = new MouselabBlock
    blockName: 'train_basic'
    stateDisplay: 'always'
    prompt: ->
      markdown """
      ## Web of Cash

      In this HIT, you will play a game called *Web of Cash*. You will guide a
      money-loving spider through a spider web. When you land on a gray circle
      (a ***node***) the value of the node is added to your score.

      You can move the spider with the arrow keys, but only in the direction
      of the arrows between the nodes. Go ahead, try it out!
    """
    lowerMessage: 'Move with the arrow keys.'
    timeline: getTrials 5
    
  train_basic2 = new MouselabBlock
    blockName: 'train_basic2'
    stateDisplay: 'always'
    prompt: ->
      markdown """
      ## Some nodes are more important than others

      #{nodeValuesDescription} Please take a look at the example below to see what this means.

      Try a few more rounds now!
    """
    lowerMessage: 'Move with the arrow keys.'
    timeline: getTrials 5

  
  train_hidden = new MouselabBlock
    blockName: 'train_hidden'
    stateDisplay: 'never'
    prompt: ->
      markdown """
      ## Hidden Information

      Nice job! When you can see the values of each node, it's not too hard to
      take the best possible path. Unfortunately, you can't always see the
      value of the nodes. Without this information, it's hard to make good
      decisions. Try completing a few more rounds.
    """
    lowerMessage: 'Move with the arrow keys.'
    timeline: getTrials 5

  train_inspector = new MouselabBlock
    blockName: 'train_inspector'
    # special: 'trainClick'
    stateDisplay: 'click'
    stateClickCost: 0
    prompt: ->
      markdown """
      ## Node Inspector

      It's hard to make good decision when you can't see what you're doing!
      Fortunately, you have access to a ***node inspector*** which can reveal
      the value of a node. To use the node inspector, simply click on a node.
      **Note:** you can only use the node inspector when you're on the first
      node.

      Trying using the node inspector on a few nodes before making your first
      move.
    """
    # but the node inspector takes some time to work and you can only inspect one node at a time.
    timeline: getTrials 5
    # lowerMessage: "<b>Click on the nodes to reveal their values.<b>"


  train_inspect_cost = new MouselabBlock
    blockName: 'train_inspect_cost'
    stateDisplay: 'click'
    stateClickCost: PARAMS.inspectCost
    prompt: ->
      markdown """
      ## The price of information

      You can use node inspector to gain information and make better
      decisions. But, as always, there's a catch. Using the node inspector
      costs $#{PARAMS.inspectCost} per node. To maximize your score, you have
      to know when it's best to gather more information, and when it's time to
      act!
    """
    timeline: getTrials 5


  bonus_text = (long) ->
    # if PARAMS.bonusRate isnt .01
    #   throw new Error('Incorrect bonus rate')
    s = "**you will earn 1 cent for every $5 you make in the game.**"
    if long
      s += " For example, if your final score is $1000, you will receive a bonus of $2."
    return s


  train_final = new MouselabBlock
    blockName: 'train_final'
    stateDisplay: 'click'
    stateClickCost: PARAMS.inspectCost
    prompt: ->
      markdown """
      ## Earn a Big Bonus

      Nice! You've learned how to play *Web of Cash*, and you're almost ready
      to play it for real. To make things more interesting, you will earn real
      money based on how well you play the game. Specifically,
      #{bonus_text('long')}

      These are the **final practice rounds** before your score starts counting
      towards your bonus.
    """
    lowerMessage: fullMessage
    timeline: getTrials 5


  train = new Block
    training: true
    timeline: [
      train_basic1
      divider    
      train_basic2    
      divider
      train_hidden
      divider
      train_inspector
      divider
      train_inspect_cost
      divider
      train_final
    ]

  quiz = new Block
    preamble: -> markdown """
      # Quiz

    """
    type: 'survey-multi-choice'
    questions: [
      "What is the range of node values in the first step?"
      "What is the range of node values in the last step?"
      "What is the cost of clicking?"
      "How much REAL money do you earn?"
    ]
    options: [
      ['$-4 to $4', '$-8 to $8', '$-48 to $48'],
      ['$-4 to $4', '$-8 to $8', '$-48 to $48'],
      ['$0', '$1', '$8', '$24'],    
      ['1 cent for every $1 you make in the game',
       '1 cent for every $5 you make in the game',
       '5 cents for every $1 you make in the game',
       '5 cents for every $10 you make in the game']
    ]

  pre_test = new ButtonBlock
    stimulus: ->
      SCORE = 0
      prompt: ''
      psiturk.finishInstructions()
      markdown """
      # Training Completed

      Well done! You've completed the training phase and you're ready to
      play *Web of Cash* for real. You will have **#{test.timeline.length}
      rounds** to make as much money as you can. Remember, #{bonus_text()}

      One more thing: **You must spend *at least* 7 seconds on each round.**
      If you finish a round early, you'll have to wait until 7 seconds have
      passed.

      To thank you for your work so far, we'll start you off with **$50**.
      Good luck!
    """


  test = new MouselabBlock
    minTime: 7
    blockName: 'test'
    stateDisplay: 'click'
    stateClickCost: PARAMS.inspectCost
    timeline: getTrials (if DEBUG then 3 else 30)
    startScore: 50
    
  verbal_responses = new Block
    type: 'survey-text'
    preamble: -> markdown """
        # Please answer these questions

      """

    questions: [
        'How did you decide where to click??'
        'How did you decide where NOT to click?'
        'How did you decide when to stop clicking?'
        'Where were you most likely to click at the beginning of each round?'
        'Can you describe anything else about your strategy?'
    ]
    button: 'Finish'

  # TODO: ask about the cost of clicking
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
      'What is your age?'
      'Additional coments?'
    ]
    button: 'Submit HIT'

  talk_demo = new Block
    timeline: [
      new MouselabBlock
        lowerMessage: 'Move with the arrow keys.'
        stateDisplay: 'always'
        prompt: null
        stateClickCost: PARAMS.inspectCost
        timeline: getTrials 3

      divider

      new MouselabBlock
        stateDisplay: 'click'
        prompt: null
        stateClickCost: PARAMS.inspectCost
        timeline: TRIALS.slice(10,14)
    ]


  if DEBUG
    experiment_timeline = [
      train
      # quiz
      pre_test
      test
      verbal_responses
      finish
    ]
  else if TALK
    experiment_timeline = [
      talk_demo
    ]
  else
    experiment_timeline = [
      train
      quiz
      pre_test
      test
      verbal_responses
      finish
    ]

  # ================================================ #
  # ========= START AND END THE EXPERIMENT ========= #
  # ================================================ #

  # bonus is the total score multiplied by something
  calculateBonus = ->
    bonus = SCORE * PARAMS.bonusRate
    bonus = (Math.round (bonus * 100)) / 100  # round to nearest cent
    return Math.max(0, bonus)
  

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

