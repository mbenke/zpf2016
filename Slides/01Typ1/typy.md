% Zaawansowane programowanie funkcyjne
% Marcin Benke
% 1 marca 2016

<meta name="duration" content="80" />

# Plan wykładu
* Typy i klasy
    * Typy algebraiczne i klasy typów
    * Klasy konstruktorowe
    * Klasy wieloparametrowe, zależności funkcyjne
* Testowanie (QuickCheck)
* Typy zależne, Agda, Idris, Coq, dowodzenie własności (ok. 7 wykladów)
* Typy zależne w Haskellu
    * Rodziny typów, typy skojarzone, uogólnione typy algebraiczne   (GADT)
	* data kinds, kind polymorphism
* Metaprogramowanie
* Programowanie równoległe w Haskellu
    * Programowanie wielordzeniowe i wieloprocesorowe (SMP)
    * Równoległość danych (Data Parallel Haskell)
* Prezentacje projektów

Jakieś życzenia?

# Zasady zaliczania
* Laboratorium: zdefiniowane zadanie Coq + prosty projekt 1-3 osobowy -Haskell
* Egzamin ustny, którego istotną częścią jest prezentacja projektu.
* Alternatywna forma zaliczenia: referat (koniecznie ciekawy!)
* ...możliwe  także inne formy.

# Materiały

~~~~~
$ cabal update
$ cabal install pandoc
$ PATH=~/.cabal/bin:$PATH            # Linux
$ PATH=~/Library/Haskell/bin:$PATH   # OS X
$ git clone git://github.com/mbenke/zpf2013.git
$ cd zpf2013/Slides
$ make
~~~~~

# języki funkcyjne
* typowane dynamicznie, gorliwe: Lisp
* typowane statycznie, gorliwe, nieczyste: ML
* typowane statycznie, leniwe, czyste: Haskell

Ten wykład: Haskell, ze szczególnym naciskiem na typy.

Bogata struktura typów jest tym, co wyróżnia Haskell wśród innych języków.

# Typy jako język specyfikacji

Typ funkcji często specyfikuje nie tylko jej wejście i wyjście ale i relacje między nimi:

~~~~ {.haskell}
f :: forall a. a -> a
f x = ?
~~~~

Jeśli `(f x)` daje wynik, to musi nim być `x`

* Philip Wadler "Theorems for Free"

* Funkcja typu `a -> IO b` może mieć efekty uboczne

    ~~~~ {.haskell}
    import Data.IORef

    f :: Int -> IO (IORef Int)
    f i = do
      print i
      r <- newIORef i
      return r

    main = do
      r <- f 42
      j <- readIORef r
      print j    
    ~~~~



# Typy jako język specyfikacji (2)

Funkcja typu `Integer -> Integer` zasadniczo nie może mieć efektów ubocznych

Liczby Fibonacciego w stałej pamięci

~~~~ {.haskell}
import Control.Monad.ST
import Data.STRef
fibST :: Integer -> Integer
fibST n = 
    if n < 2 then n else runST fib2 where
      fib2 =  do
        x <- newSTRef 0
        y <- newSTRef 1
        fib3 n x y
 
      fib3 0 x _ = readSTRef x
      fib3 n x y = do
              x' <- readSTRef x
              y' <- readSTRef y
              writeSTRef x y'
              writeSTRef y (x'+y')
              fib3 (n-1) x y
~~~~

Jak to?

~~~~
runST :: (forall s. ST s a) -> a
~~~~

Typ `runST` gwarantuje, że efekty uboczne nie wyciekają. Funkcja `fibST`
jest czysta.

# Typy jako język projektowania

* Projektowanie programu przy użyciu typów i `undefined`

    ~~~~ {.haskell}
    conquer :: [Foo] -> [Bar]
    conquer fs = concatMap step fs

    step :: Foo -> [Bar]
    step = undefined
    ~~~~

# Typy jako język programowania

*    Funkcje na typach obliczane w czasie kompilacji

    ~~~~ {.haskell}
    data Zero
    data Succ n

    type One   = Succ Zero
    type Two   = Succ One
    type Three = Succ Two
    type Four  = Succ Three

    one   = undefined :: One
    two   = undefined :: Two
    three = undefined :: Three
    four  = undefined :: Four

    class Add a b c | a b -> c where
      add :: a -> b -> c
      add = undefined
    instance              Add  Zero    b  b
    instance Add a b c => Add (Succ a) b (Succ c)
    ~~~~ 

    ~~~~
    *Main> :t add three one
    add three one :: Succ (Succ (Succ (Succ Zero)))
    ~~~~

* Ćwiczenie: rozszerzyć o mnożenie i silnię

# Typy jako język programowania (2)
Wektory przy użyciu klas:

~~~~ {.haskell}
data Vec :: * -> * -> * where
  VNil :: Vec Zero a  
  (:>) :: a -> Vec n a -> Vec (Succ n) a

vhead :: Vec (Succ n) a -> a
vhead (x :> xs) = x
~~~~

**Ćwiczenie:** dopisać `vtail`, `vlast`

Chcielibyśmy również mieć

~~~~ {.haskell}
vappend :: Add m n s => Vec m a -> Vec n a -> Vec s a
~~~~

ale tu niestety system typów okazuje się za słaby

# Typy jako język programowania (3)

* Wektory przy użyciu rodzin typów:

    ~~~~ {.haskell}
    data Zero = Zero
    data Suc n = Suc n

    type family m :+ n
    type instance Zero :+ n = n
    type instance (Suc m) :+ n = Suc(m:+n)

    data Vec :: * -> * -> * where
      VNil :: Vec Zero a  
      (:>) :: a -> Vec n a -> Vec (Suc n) a

    vhead :: Vec (Suc n) a -> a
    vappend :: Vec m a -> Vec n a -> Vec (m:+n) a
    ~~~~

* Sprytna sztuczka o wątpliwej wartości praktycznej

# Typy zależne

Prawdziwe programowanie na poziomie typów  i dowodzenie własności programów możliwe w języku z typami zależnymi, takim jak Agda, Epigram, Idris

~~~~
module Data.Vec where
infixr 5 _∷_

data Vec (A : Set a) : ℕ → Set where
  []  : Vec A zero
  _∷_ : ∀ {n} (x : A) (xs : Vec A n) → Vec A (suc n)

_++_ : ∀ {a m n} {A : Set a} → Vec A m → Vec A n → Vec A (m + n)
[]       ++ ys = ys
(x ∷ xs) ++ ys = x ∷ (xs ++ ys)

module UsingVectorEquality {s₁ s₂} (S : Setoid s₁ s₂) where
  xs++[]=xs : ∀ {n} (xs : Vec A n) → xs ++ [] ≈ xs
  xs++[]=xs []       = []-cong
  xs++[]=xs (x ∷ xs) = SS.refl ∷-cong xs++[]=xs xs
~~~~


# Problem z typami zależnymi

O ile Haskell bywa czasami nieczytelny, to z typami zależnymi całkiem łatwo przesadzić:

~~~~
  now-or-never : Reflexive _∼_ →
                 ∀ {k} (x : A ⊥) →
                 ¬ ¬ ((∃ λ y → x ⇓[ other k ] y) ⊎ x ⇑[ other k ])
  now-or-never refl x = helper <$> excluded-middle
    where
    open RawMonad ¬¬-Monad

    not-now-is-never : (x : A ⊥) → (∄ λ y → x ≳ now y) → x ≳ never
    not-now-is-never (now x)   hyp with hyp (, now refl)
    ... | ()
    not-now-is-never (later x) hyp =
      later (♯ not-now-is-never (♭ x) (hyp ∘ Prod.map id laterˡ))

    helper : Dec (∃ λ y → x ≳ now y) → _
    helper (yes ≳now) = inj₁ $ Prod.map id ≳⇒ ≳now
    helper (no  ≵now) = inj₂ $ ≳⇒ $ not-now-is-never x ≵now
~~~~

...chociaż oczywiście pisanie takich dowodów jest świetną zabawą.


# Data Parallel Haskell

Dokąd chcemy dojść: 

~~~~ {.haskell}
{-# LANGUAGE ParallelArrays #-}
{-# OPTIONS_GHC -fvectorise #-}

module DotP where
import qualified Prelude
import Data.Array.Parallel
import Data.Array.Parallel.Prelude
import Data.Array.Parallel.Prelude.Double as D

dotp_double :: [:Double:] -> [:Double:] -> Double
dotp_double xs ys = D.sumP [:x * y | x <- xs | y <- ys:]
~~~~

Wygląda jak operacja na listach, ale działa na tablicach i
"automagicznie" zrównolegla się na dowolną liczbę rdzeni/procesorów
(także CUDA).

Po drodze czeka nas jednak trochę pracy.

# Typy w Haskellu

* typy bazowe: `zeroInt :: Int`
* typy funkcyjne: `plusInt :: Int -> Int -> Int`
* typy polimorficzne `id :: a -> a`

    ~~~~ {.haskell}
    {-# LANGUAGE ExplicitForAll #-}
    g :: forall b.b -> b
    ~~~~

* typy algebraiczne 

    ~~~~ {.haskell}
    data Tree a = Leaf | Node a (Tree a) (Tree a)
    ~~~~

* `Leaf` i `Node` są konstruktorami wartości: 

    ~~~~ {.haskell}
    data Tree a where
    	 Leaf :: Tree a
         Node :: a -> Tree a -> Tree a -> Tree a
    ~~~~

* `Tree` jest *konstruktorem typowym*, czyli operacją na typach

* NB od niedawna Haskell dopuszcza puste typy:

    ~~~~ {.haskell}
    data Zero
    ~~~~
  
# Typowanie polimorficzne

* Generalizacja:

$${\Gamma \vdash e :: t, a \notin FV( \Gamma )}\over {\Gamma \vdash e :: \forall a.t}$$

 <!-- 
Jeśli $\Gamma \vdash e :: t, a \notin FV( \Gamma )$
 
to $\Gamma \vdash e :: \forall a.t$

  Γ ⊢ e :: t, a∉FV(Γ)
$$\Gamma \vdash e :: t$$ ,
 \(a \not\in FV(\Gamma) \) , 
to $\Gamma \vdash e :: \forall a.t$
-->

Na przykład

$${ { \vdash map :: (a\to b) \to [a] \to [b] } \over
   { \vdash map :: \forall b. (a\to b) \to [a] \to [b] } } \over
   { \vdash map :: \forall a. \forall b. (a\to b) \to [a] \to [b] } $$

Uwaga:

$$ f : a \to b \not \vdash map\; f :: \forall b. [a] \to [b]  $$

* Instancjacja

$$ {\Gamma \vdash e :: \forall a.t}\over {\Gamma \vdash e :: t[a:=s]} $$
 
# Klasy

* klasy opisują własności typów

    ~~~~ {.haskell}
    class Eq a where
      (==) :: a -> a -> Bool
    instance Eq Bool where
       True  == True  = True
       False == False = True
       _     == _     = False
    ~~~~
    
* funkcje mogą być definiowane w kontekście klas:

    ~~~~ {.haskell}
    elem :: Eq a => a -> [a] -> Bool
    ~~~~

+ Implementacja 
    - instancja tłumaczona na słownik metod (coś \'a la  vtable w C++)
    - kontekst (np Eq a) jest tłumaczony na ukryty parametr (słownik metod )
    - podklasa tłumaczona na funkcję


# Operacje na typach

* Prosty przykład:

    ~~~~ {.haskell}
    data Tree a = Leaf | Node a (Tree a) (Tree a)
    ~~~~

* Konstruktory typowe transformują typy

* `Tree` może zamienić np. `Int` w drzewo

+ Funkcje wyższego rzędu transformują funkcje

+ Konstruktory wyższego rzędu transformują konstruktory typów

~~~~ {.haskell}
newtype IdentityT m a = IdentityT { runIdentityT :: m a }
~~~~ 

# Klasy konstruktorowe

* klasy konstruktorowe opisują własności konstruktorów typów:

    ~~~~ {.haskell}
    class Functor f where
      fmap :: (a->b) -> f a -> f b
    (<$>) = fmap

    instance Functor [] where
      fmap = map

    class Functor f => Pointed f where
       pure :: a -> f a
    instance Pointed [] where
       pure = (:[])

    class Pointed f => Applicative f where
      (<*>) :: f(a->b) -> f a -> f b 

    instance Applicative [] where
      fs <*> xs = concat $ flip map fs (flip map xs)

    class Applicative m => Monad' m where
      (>>=) :: m a -> (a -> m b) -> m b
    ~~~~

<!-- 

    class Pointed f => Applicative f where
      (<*>) :: f(a->b) -> f a -> f b 
      (*>) :: f a -> f b -> f b
      x *> y = (flip const) <$> x <*> y
      (<*) :: f a -> f b -> f a
      x <* y = const <$> x <*> y

    liftA2 :: Applicative f => (a -> b -> c) -> f a -> f b -> f c
    liftA2 f a b = f <$> a <*> b

-->

# Rodzaje (kinds)

* Operacje na wartościach są opisywane przez ich typy

* Operacje na typach są opisywane przez ich rodzaje (kinds)

* Typy (np. `Int`) są rodzaju `*`

* Jednoargumentowe konstruktory (np. `Tree`) są rodzaju `* -> *`

    ~~~~ {.haskell}
    {-#LANGUAGE KindSignatures, ExplicitForAll #-}

    class Functor f => Pointed (f :: * -> *) where
        pure :: forall (a :: *).a -> f a
    ~~~~

* Występują też bardziej złożone rodzaje, np. dla transformatorów monad:

    ~~~~ {.haskell}
    class MonadTrans (t :: (* -> *) -> * -> *) where
        lift :: Monad (m :: * -> *) => forall (a :: *).m a -> t m a
    ~~~~

NB spacje są niezbędne - `::*->*` jest jednym leksemem.

# Klasy wieloparametrowe

* Czasami potrzebujemy opisać nie tyle pojedynczy typ, co relacje między typami:

    ~~~~ {.haskell}
    {-#LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
    class Iso a b where
      iso :: a -> b
      osi :: b -> a
      
    instance Iso a a where
      iso = id
      osi = id

    instance Iso ((a,b)->c) (a->b->c) where
      iso = curry
      osi = uncurry

    instance (Iso a b) => Iso [a] [b] where
     iso = map iso
     osi = map osi
    ~~~~

* Uwaga: w ostatnim przykładzie `iso` ma inny typ po lewej, inny po prawej 

* Ćwiczenie: napisz jeszcze jakieś instancje klasy `Iso`


    ~~~~ {.haskell}
    instance (Functor f, Iso a b) => Iso (f a) (f b) where 
    instance Iso (a->b->c) (b->a->c) where
    ~~~~

# Dygresja - FlexibleInstances

Haskell 2010

<!--
An instance declaration introduces an instance of a class. Let class
cx => C u where { cbody } be a class declaration. The general form of
the corresponding instance declaration is: instance cx′ => C (T u1 …
uk) where { d } where k ≥ 0. The type (T u1 … uk) must take the form
of a type constructor T applied to simple type variables u1, … uk;
furthermore, T must not be a type synonym, and the ui must all be
distinct.
-->

* an instance head must have the form C (T u1 ... uk), where T is a type constructor defined by a data or newtype declaration  and the ui are distinct type variables, and

<!--
*    each assertion in the context must have the form C' v, where v is one of the ui. 
-->

This prohibits instance declarations such as:

  instance C (a,a) where ...  
  instance C (Int,a) where ...  
  instance C [[a]] where ...

`instance Iso a a` nie spełnia tych warunków, ale wiadomo o jaką relację nam chodzi :)

# Problem z klasami wieloparametrowymi
Spróbujmy stworzyć klasę kolekcji, np.

`BadCollection.hs`

~~~~ {.haskell}
class Collection c where
  insert :: e -> c -> c    
  member :: e -> c -> Bool

instance Collection [a] where
     insert = (:)
     member = elem  
~~~~

~~~~
    Couldn't match type `e' with `a'
      `e' is a rigid type variable bound by
          the type signature for member :: e -> [a] -> Bool
          at BadCollection.hs:7:6
      `a' is a rigid type variable bound by
          the instance declaration
          at BadCollection.hs:5:22
~~~~

Dlaczego?

# Problem z klasami wieloparametrowymi

~~~~ {.haskell}
class Collection c where
 insert :: e -> c -> c    
 member :: e -> c -> Bool
~~~~

tłumaczy się (mniej więcej) do

~~~~
data ColDic c = CD 
 { 
 , insert :: forall e.e -> c -> c
 , member :: forall e.e -> c -> Bool
 }
~~~~

 ... nie o to nam chodziło.
   
~~~~ {.haskell}
instance Collection [a] where
   insert = (:)
   member = undefined
~~~~

~~~~
-- (:) :: forall t. t -> [t] -> [t]
ColList :: forall a. ColDic a
ColList = \@ a -> CD { insert = (:) @ a, member = 
~~~~

# Problem z klasami wieloparametrowymi

 <!--- `BadCollection2.hs` -->
<!---
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
-->

~~~~ {.haskell}
class Collection c e where
  insert :: e -> c -> c
  member :: e -> c -> Bool

instance Eq a => Collection [a] a where
  insert  = (:)
  member = elem
     
ins2 x y c = insert y (insert x c)
-- ins2 :: (Collection c e, Collection c e1) => e1 -> e -> c -> c

problem1 :: [Int]
problem1 = ins2 1 2 []
-- No instances for (Collection [Int] e0, Collection [Int] e1)
-- arising from a use of `ins2'

problem2 = ins2 'a' 'b' []
-- No instance for (Collection [a0] Char)
--       arising from a use of `ins2'

problem3 :: (Collection c0 Char, Collection c0 Bool) => c0 -> c0
problem3 = ins2 True 'a'
-- Tu problem akurat polega na tym, że to jest poprawne typowo
-- ...a chyba nie powinno być
~~~~

# Zależności funkcyjne
Czasami w klasach wieloparametrowych, jeden parametr wyznacza inny, np.

~~~~ {.haskell}
 class (Monad m) => MonadState s m | m -> s where ...

 class Collects e ce | ce -> e where
      empty  :: ce
      insert :: e -> ce -> ce
      member :: e -> ce -> Bool
~~~~

Problem: *Fundeps are very, very tricky.* - SPJ

Więcej: http://research.microsoft.com/en-us/um/people/simonpj/papers/fd-chr/

# Refleksja - czemu nie klasy konstruktorowe?

Problem kolekcji możemy rozwiązać np. tak:

~~~~ {.haskell}
class Collection c where
  insert :: e -> c e -> c e
  member :: Eq e => e -> c e-> Bool

instance Collection [] where
     insert x xs = x:xs
     member = elem
~~~~

ale nie rozwiązuje to problemu np. z monadą stanu:

~~~~ {.haskell}
 class (Monad m) => MonadState s m | m -> s where 
   get :: m s
   put :: s -> m ()
~~~~

typ stanu nie jest tu parametrem konstruktora m.

# Fundeps are very very tricky

~~~~ {.haskell}
class Mul a b c | a b -> c where
  (*) :: a -> b -> c
  
newtype Vec a = Vec [a]
instance Functor Vec where
  fmap f (Vec as) = Vec $ map f as
  
instance Mul a b c => Mul a (Vec b) (Vec c) where
  a * b = fmap (a*) b
  
f t x y = if t then  x * (Vec [y]) else y
~~~~

Jakiego typu jest f? Niech x::a, y::b. 

Wtedy typem wyniku jest b i musimy mieć instancję `Mul a (Vec b) b`

Z kolei `a b -> c` implikuje, że `b = Vec c` dla pewnego c, czyli szukamy instancji

~~~~
Mul a (Vec (Vec c)) (Vec c)
~~~~

zastosowanie reguły `Mul a b c => Mul a (Vec b) (Vec c)` doprowadzi nas do `Mul a (Vec c) c`.

...i tak w kółko.


# Spróbujmy

~~~~ {.haskell}
Mul1.hs:16:21:
    Context reduction stack overflow; size = 21
    Use -fcontext-stack=N to increase stack size to N
      co :: c18 ~ Vec c19
      $dMul :: Mul a0 c17 c18
      $dMul :: Mul a0 c16 c17
      ...
      $dMul :: Mul a0 c1 c2
      $dMul :: Mul a0 c c1
      $dMul :: Mul a0 c0 c
      $dMul :: Mul a0 (Vec c0) c0
    When using functional dependencies to combine
      Mul a (Vec b) (Vec c),
        arising from the dependency `a b -> c'
        in the instance declaration at 3/Mul1.hs:13:10
      Mul a0 (Vec c18) c18,
        arising from a use of `mul' at 3/Mul1.hs:16:21-23
    In the expression: mul x (Vec [y])
    In the expression: if b then mul x (Vec [y]) else y
~~~~

(musimy użyć UndecidableInstances, żeby GHC w ogóle spróbowało - ten przykład pokazuje co jest 'Undecidable').

# Rodziny typów

Rodziny to funkcje na typach - jak na pierwszym wykładzie

~~~~ {.haskell}
{-# TypeFamilies #-}

data Zero = Zero
data Suc n = Suc n

type family m :+ n
type instance Zero :+ n = n
type instance (Suc m) :+ n = Suc(m:+n)

vhead :: Vec (Suc n) a -> a
vappend :: Vec m a -> Vec n a -> Vec (m:+n) a
~~~~

Trochę dalej powiemy sobie o nich bardziej systematycznie.
