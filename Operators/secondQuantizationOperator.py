from BinaryOperators import addParticle,removeParticle,sign as addParticle,removeParticle,sign


def secondQuantizationOneBodyOperator(p,q,state1,state2):
    phase = 1

    removeParticle(q,state1)
    if state1(-1) == True:
        return 0
    phase *= sign(q,state1)

    addParticle(p,state1)
    if state1(-1)== True:
        return 0
    phase *= sign(p,state1)

    if state2 != state1:
        phase = 0
    return phase

def secondQuantizationTwoBodyOperator(p,q,r,s,state1,state2):
    phase = 1
    #  Removing r
    removeParticle(r,state1)
    if state1(-1)== True:
        return 0
    phase *= sign(r,state1)
    # Removing s
    removeParticle(s,state1)
    if state1(-1)== True:
        return 0
    phase *= sign(s,state1)
    # Adding q
    addParticle(q,state1)
    if state1(-1)== True:
        return 0
    phase *= sign(q,state1)
    # Adding p
    addParticle(p,state1)
    if state1(-1)== True:
        return 0
    phase *= sign(p,state1)
    if state1 != state2:
        return 0
    return phase