from toolz import memoize, compose

h = compose(hash, str)
def hash_312(state):
    if state == '__term_state__':
        return hash(state)
    s = [hash(x) + 100000 for x in state]
    return (
      h(s[1] + 
        h(s[2] +
          h(s[3]) +
          h(s[4]))
      ) +
      h(s[5] + 
        h(s[6] +
          h(s[7]) +
          h(s[8]))
      ) + 
      h(s[9] + 
        h(s[10] +
          h(s[11]) +
          h(s[12]))
      )
    )

def solve(env, cache=None, hash_state=hash_312):
    """Returns Q, V, pi, and computation data for an mdp environment."""

    info = {  # track number of times each function is called
        'q': 0,
        'v': 0
    }
    if hash_state is not None:
        def hash_key(args, kwargs):
            state = args[0]
            if state is None:
                return state
            else:
                return hash_state(state)
    else:
        hash_key = None

    def Q(s, a):
        info['q'] += 1
        return sum(p * (r + V(s1)) for p, s1, r in env.results(s, a))

    @memoize(key=hash_key, cache=cache)
    def V(s):
        if s is None:
            return 0
        info['v'] += 1
        return max((Q(s, a) for a in env.actions(s)), default=0)
    
    def pi(s):
        return max(env.actions(s), key=lambda a: Q(s, a))
    
    return Q, V, pi, info



