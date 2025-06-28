---
layout: post
title:  Notes on "Future of Decentralization, AI, and Computing Summit"
date:   2023-09-08 16:03:28 -0700
description: Summary of insights from Berkeley RDI's summit, focusing on Noam Brown's presentation about using reinforcement learning to play Diplomacy and the challenges of applying RL to cooperative games.
tags: [ml]
math: true
---

## Introduction
I recently went to the "Future of Decentralization, AI, and Computing Summit"
hosted by Berkeley RDI. One of the most compelling presentations was Noam
Brown's presentation about using reinforcement learning to play the game of
[Diplomacy](https://en.wikipedia.org/wiki/Diplomacy_(game)).

This is of particular intrest due to Diplomacy's large focus on cooperation.
Most current algorithms designed for playing games will not work at for
Diplomacy.

To start off, we will go over a short background of current game playing
algorithms and explain their issue on cooperative games such as Diplomacy.

### What's RL?
Recently, games such as Chess and Go have become dominated by superhuman play by
computers. How are these powered? The answer is reinforcement learning.

Reinforcement learning (rl) is a subset of machine learning which focuses on
learning from the enviornment - without human labeled information. A rl agent
must go through an enviornment on its own and learn from its own experiences
what the best actions in each situation are. More formally, a rl agent maps from
situations to actions (Sutton et. al).

To do this the agent must both explore its enviornment to see what it can expect
from certain actions, while also exploiting the best ones - improving what seems
most promising. This problem is exceedingly complicated and still remains an
open problem.

How an agent is implemented and chooses when to explore is beyond the scope of
this post, but several resources that could help can be found at the end. 

### 2 Player Zero Sum Games
One class of problems that rl excels in are 2 player zero sum games. What are
these?

Well clearly a 2 player 0 sum game requires two players. This means games like
Chess, Go, and 2 player poker. Now what about 0 sum? A zero sum game is a game
in which one player's gain is equal to the opponent's loss. This means that
there is no additional benefits that will appear.

For example, in Chess, all the gains in one player's position mean's that the
opponent's will be worse by that same amount. Similarly, in poker all of one
players gains equal everyone elses losses.

In rl this is important because of the concept of minimax. Minimax is the
concept that you will play the best moves assuming your oppoenent also playes
the best moves. This ensures that, on average, you will not lose. For example, a
poker minimax strategy will never lose in the long run.

We can find the minimax equilibrium through self play. Our rl agent will play
against itself until it learns this optimal strategy. Assuming we have written
it well with enough exploration, if it plays for long enough then it is
garunteed to find the minimax equilibrium. To make sense of why, assume there is
some other strategy that will win against our current one, eventually we would
explore into it (assuming we did everything correctly) and use that better
strategy. This process would continue and over infinite time, we would explore
every strategy and settle upon the best. Any strategy that would exploit ours
would have been found.

However, this system quickly falls apart when it is no longer a 2 player zero
sum game.

### Ultimatum
The simplist example for this is Ultimatum. Ultimatum is a game with 2 players,
one starts off with a hundred dollars and can choose to offer any amount to the
other player. The other player can either accept what is offered or reject it.
If they accept they split the hundred at the agreed ratio and keep it. However,
if they reject, both players end up with zero.

This is not a zero sum game because if the other player rejects the offer, then
both players end up with 0, changing the total money in the system.

So what would self play learn? Self play would simply learn to offer the minimal
amount of money possible as that is theoretically the best way to play. Both
player 1 and 2 end up with more than the alternative so clearly its the optimal
outcome. However this approach does not work when it comes to humans. Almost all
people would reject this offer. Clearly the rl agent playing game theory
optimally would fail. This is because humans are irrational, so this game won't
ever work through self play. No rl agent playing against itself only would learn
human preferences.

How do we fix this? The only way to fix this is with human data. The agent would
either need to play against humans or see pure human games and learn from those.
Without this there is no way for the agent to adapt to human irrationality.

## Diplomacy
For a more complex example, we can look at Diplomacy. Diplomacy is a old game
from the 50s. It attempts to simulate the alliances in WW1. This means that it
requires cooperation between different players (the alliances), but also
eventually requiring betraying allies. Each turn you communicate with other
players making promises (under no obligations to hold them) and then write your
final actual orders. Clearly to win, you require communication and the right
amount of trust as alliances are necessary to win.

To test out if self-play works, the team created a self play bot called DORA. It
used a algorithm similar to one of the top chess bots AlphaZero. Using this,
they found that when they played a modified version of diplomacy involving only
2 players, it would achieve amazing performance, winning around 86% of its
matches against human experts.

However, Diplomacy is meant to be played with 7 players. When placing 6 DORA
agents against 1 human like agent (one trained to mimick how humans play), the
human like agent only won around 1% of the time. These results however, don't
hold in the inverse situation. When 1 DORA played against 6 human like bots,
it's performance dropped to 11% (14% would be considered average). Brown notes
that this means that the DORA agents learned there own game theory optimal
language, similar to how an ultimatum agent would learn its own game theory
optimal strategy, however, when faced against irrational human like agents, this
would not work.

To get around this, they introduced piKL. How piKL fixes the issue of self play
finding strategies, while game theory optimal, that won't work against humans is
by attempting to make the self play strategies remain similar to human ones. It
does this by first training a agent on purely human data. Then, using self play
it learns its own policy which gets penalized for straying to far away from the
human strategy.

This leads to getting closer to the optimal strategy while still being able to
work with humans. The Diplomacy agent trained with this strategy found far more
success, placing in the top 10% of humans in Diplomacy.
## References
Sutton et. al
Brown et. al
