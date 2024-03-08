/// Removes e from v if e exists in v; otherwise leaves v untouched.
pub fn remove<T>(v: &mut Vec<T>, e: T)
where
    T: PartialEq,
{
    if let Some(i) = v.iter().position(|x| *x == e) {
        v.remove(i);
    }
}

/// Creates a vector of the specified size, with the provided defaults.
pub fn vec_with_size<T>(sz: usize, default: T) -> Vec<T>
where
    T: Clone,
{
    let mut v = Vec::with_capacity(sz);
    (0..sz).for_each(|_| v.push(default.clone()));
    v
}
