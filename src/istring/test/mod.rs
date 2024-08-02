use super::{CachedString, IStringStore, GLOBAL_STORE};

#[test]
fn interning_empty_string() {
  let empty_string = "";

  let tok = empty_string.intern();
  "".to_string().intern();

  assert_eq!(tok.to_string(), "");
  assert_eq!(tok.to_string().as_str(), "");
  assert_eq!(GLOBAL_STORE._data.read().unwrap().is_empty(), true);
}

#[test]
fn interning_small_string() {
  let small_string = "test";

  let tok = small_string.intern();

  assert_eq!(tok.to_string(), "test");
  assert_eq!(tok.to_string().as_str(), "test");
  assert_eq!("B".intern().to_string(), "B");
  assert_eq!("B".intern().to_str().as_str(), "B");
  assert!("function".intern().is_small());
  assert_eq!("function".intern().to_str().as_str(), "function");
}

#[test]
fn interning_large_string() {
  let large_string = "
  Lumina eiusdem a sororibus est agant montis tu urbes succedit gavisa dolore
Perseus incerti, repente pariter. Omnes morsu rediit flores, nisi scelus
confessis cristati ramis silentum arentis centimanum sacrilegae pone. Silvas
dieque ire *fuit*, resides videt quodcumque illi circumflua petii laeva ne
timidas Venilia. Accipiunt saltem apta modo! Annos tu tale concita nostro
relicto orbe quid, adit nec possederat simque conprendere defecerat avus
Cyparisse multarum!";

  let tok = large_string.intern();

  assert_eq!(tok.to_string(), large_string);
  assert_eq!(tok.to_str().as_str(), large_string);
}

#[test]
fn interning_same_large_string() {
  let large_str = "
  Lumina eiusdem a sororibus est agant montis tu urbes succedit gavisa dolore
Perseus incerti, repente pariter. Omnes morsu rediit flores, nisi scelus
confessis cristati ramis silentum arentis centimanum sacrilegae pone. Silvas
dieque ire *fuit*, resides videt quodcumque illi circumflua petii laeva ne
timidas Venilia. Accipiunt saltem apta modo! Annos tu tale concita nostro
relicto orbe quid, adit nec possederat simque conprendere defecerat avus
Cyparisse multarum!";

  let string = String::from(large_str);

  let tok_a = large_str.intern();
  let tok_b = string.intern();

  assert_eq!(tok_a.to_string(), large_str);
  assert_eq!(tok_b.to_str().as_str(), large_str);
  assert_eq!(tok_a, tok_b);
}

#[test]
fn interning_different_large_string() {
  let large_str_a = "1: Lumina eiusdem a sororibus est agant montis tu urbes succedit gavisa dolore
Perseus incerti, repente pariter. Omnes morsu rediit flores, nisi scelus
confessis cristati ramis silentum arentis centimanum sacrilegae pone. Silvas
dieque ire *fuit*, resides videt quodcumque illi circumflua petii laeva ne
timidas Venilia. Accipiunt saltem apta modo! Annos tu tale concita nostro
relicto orbe quid, adit nec possederat simque conprendere defecerat avus
Cyparisse multarum!";

  let large_str_b = "2: Lumina eiusdem a sororibus est agant montis tu urbes succedit gavisa dolore
Perseus incerti, repente pariter. Omnes morsu rediit flores, nisi scelus
confessis cristati ramis silentum arentis centimanum sacrilegae pone. Silvas
dieque ire *fuit*, resides videt quodcumque illi circumflua petii laeva ne
timidas Venilia. Accipiunt saltem apta modo! Annos tu tale concita nostro
relicto orbe quid, adit nec possederat simque conprendere defecerat avus
Cyparisse multarum!";

  let tok_a = large_str_a.intern();
  let tok_b = large_str_b.intern();

  assert!(tok_a.is_large());
  assert!(tok_b.is_large());
  assert_eq!(tok_a.to_string(), large_str_a);
  assert_eq!(tok_b.to_string(), large_str_b);
  assert_ne!(tok_a.to_string(), large_str_b);
  assert_ne!(tok_b.to_string(), large_str_a);
  assert_ne!(tok_a, tok_b);
  assert_eq!(GLOBAL_STORE.len(), 2);
}

#[test]
fn interning_strings_on_different_threads() {
  std::thread::scope(|scope| {
    for u in 0..4 {
      scope.spawn(move || {
        let large_str_a = "1: Lumina eiusdem a sororibus est agant montis tu urbes succedit gavisa dolore
        Perseus incerti, repente pariter. Omnes morsu rediit flores, nisi scelus
        confessis cristati ramis silentum arentis centimanum sacrilegae pone. Silvas
        dieque ire *fuit*, resides videt quodcumque illi circumflua petii laeva ne
        timidas Venilia. Accipiunt saltem apta modo! Annos tu tale concita nostro
        relicto orbe quid, adit nec possederat simque conprendere defecerat avus
        Cyparisse multarum!";

        let large_str_b = "2: Lumina eiusdem a sororibus est agant montis tu urbes succedit gavisa dolore
        Perseus incerti, repente pariter. Omnes morsu rediit flores, nisi scelus
        confessis cristati ramis silentum arentis centimanum sacrilegae pone. Silvas
        dieque ire *fuit*, resides videt quodcumque illi circumflua petii laeva ne
        timidas Venilia. Accipiunt saltem apta modo! Annos tu tale concita nostro
        relicto orbe quid, adit nec possederat simque conprendere defecerat avus
        Cyparisse multarum!";

        for i in 0..400 {
          (large_str_a.to_string() + &format!(" {i} {u}")).intern();
          (large_str_b.to_string() + &format!(" {i}")).intern();
        }
        1
      });
    }
  });

  assert!(GLOBAL_STORE.len() >= 2000);
}
